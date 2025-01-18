import os
import sys
import time
import yaml
import json
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
sys.path.append(str("src/helpers"))

from utils import (
    date_time_record,
    load_parameters,
    model_pipeline,
)
from helpers import config_loader
from helpers import create_model_comparison_plots, hyperparameter_optimization_helper, config_loader, update_configs_yml


def calculate_model_metrics(model: BaseEstimator, X, y_true):
    """Calculates and returns a dictionary of model metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    f1 = f1_score(y_true, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_true, y_prob, average="weighted", multi_class="ovo")

    metrics = {"f1": np.round(f1, 3), "roc_auc": np.round(roc_auc, 3)}
    return metrics


def aggregate_cross_validation_scores(fold_scores: list[dict]):
    """
    Aggregates scores obtained from multiple cross-validation folds."""
    collection_scores = {}
    for fold_scores_dict in fold_scores:
        for metric_name, metric_value in fold_scores_dict.items():
            if metric_name not in collection_scores:
                collection_scores[metric_name] = [metric_value]
            else:
                collection_scores[metric_name].append(metric_value)

    mean_scores = {
        metric_name: np.round(np.mean(scores), 3) 
        for metric_name, scores in collection_scores.items()
    }
    return mean_scores

def cross_valid_iteration(
    baseline: BaseEstimator,
    x: pd.DataFrame,
    y: pd.DataFrame,
    num_split: int,
    seed: float,
) -> dict:
    pipe = model_pipeline(baseline)

    label_encoding = LabelEncoder()
    y_all = label_encoding.fit_transform(y)

    # Cross-Validation
    cv = StratifiedKFold(n_splits=num_split, shuffle=True, random_state=seed)

    train_scores = []
    val_scores = []

    for idx, (train_idx, val_idx) in tqdm(enumerate(cv.split(x, y_all))):
        print(" Iteration Count:", idx + 1)
        x_train_, x_val_ = x.iloc[train_idx], x.iloc[val_idx]
        y_train_, y_val_ = y_all[train_idx], y_all[val_idx]

        pipe.fit(x_train_, y_train_)

        train_score = calculate_model_metrics(pipe, x_train_, y_train_)
        val_score = calculate_model_metrics(pipe, x_val_, y_val_)

        train_scores.append(train_score)
        val_scores.append(val_score)

    training_score = aggregate_cross_validation_scores(train_scores)
    validation_score =aggregate_cross_validation_scores(val_scores)

    metric_scores = {"train": training_score, "valid": validation_score}

    return metric_scores


if __name__ == "__main__":
    print("Started cross validation ...")
    params_loader = config_loader("configs.yml")

    # Reading data file
    training_in_ = sys.argv[1]
    cv_output_path_ = sys.argv[2]

    files_path_ = params_loader["data"]["split"]["files"][0]
    file_name_path_ = Path(training_in_) / files_path_
    os.makedirs(cv_output_path_, exist_ok=True)

    # Cross-Validation utils
    checkpoints = params_loader["models"]["baseline"]["path"]
    model_checkpoints = Path(params_loader["models"]["path"]) / checkpoints

    model_name_ = params_loader["models"]["train"]["model"]
    num_split_ = params_loader["models"]["cross_validation"]["n_split"]
    seed_ = params_loader["models"]["cross_validation"]["seed"]


    # Load training dataset
    dataframe = pd.read_csv(file_name_path_)
    x_all_ = dataframe[["text"]]
    y_all_ = dataframe["sentiment"]

    # Initiate cross validation process
    search_model_ = f"{model_name_}_model.pkl"
    
    # Check if model file exist in development stage
    try:
        models = [
            str(f).rsplit("/", maxsplit=1)[-1]
            for f in list(Path(model_checkpoints).glob("*.pkl"))
        ]

        if not search_model_ in models:
            raise ValueError(
                f"`{search_model_}` model does not exists! Please run model_dev.py script."
            )

    except ValueError as e:
        sys.exit(f"{e}")

    
    model_pickle_ = Path(model_checkpoints) / search_model_
    clf = joblib.load(model_pickle_)

    scores_ = cross_valid_iteration(clf, x_all_, y_all_, num_split_, seed_)
    score_out_ = Path(cv_output_path_) / f"metrics.json"

    with open(score_out_, "w", encoding="utf-8") as f:
        json.dump(scores_, f, ensure_ascii=False, indent=4)

    plot_file_out_ = Path(cv_output_path_) / f"cv_scores_plot.png"
    create_model_comparison_plots(scores_, plot_file_out_)

    baseline_path = Path(params_loader["models"]["path"]) / "trained/metrics.json"
    validation_path = Path(params_loader["models"]["path"]) / "cross_valid/metrics.json"

    bool_optimizer = hyperparameter_optimization_helper(baseline_path, validation_path)
    
    if bool_optimizer: 
        print("Updating train.optimize configs ...")
        update_configs_yml("configs.yml", "optimize", bool_optimizer)
    else:
        sys.exit("No need to optimize parameters.")

