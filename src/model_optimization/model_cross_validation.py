import os
import sys
import time
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

from helpers import create_model_comparison_plots, hyperparameter_optimization_helper

# Barplots for metric comparisons
def plot_cross_valid_score(scores: dict, out_path: str, img_pixel=100) -> None:
    filter_metric = []
    for c, k in scores.items():
        c_split = c.split("_")
        if "time" in c:
            continue

        if len(c_split) > 2:
            values = (c_split[0], "_".join(c_split[1:]), k)
        else:
            values = (*c_split, k)

        filter_metric.append(values)

    cols = ["fold", "metric", "score"]
    metric_df = pd.DataFrame(filter_metric, columns=cols).sort_values(
        "metric", ascending=False
    )

    # Seaborn plot
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.barplot(
        metric_df,
        x="metric",
        y="score",
        ax=ax,
        width=0.5,
        hue="fold",
        palette="colorblind",
    )
    # Axis Labels and Title
    ax.set_title("cross validation metrics", fontsize=12, alpha=0.8)
    ax.set_ylabel("score", fontsize=12)
    ax.set_xlabel("")
    plt.legend(title="")

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=img_pixel)


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
    params_loader = load_parameters("configs.yml")

    # Reading data file
    parent_split_ = params_loader["data"]
    path_split_ = parent_split_["split"]
    path_in_ = path_split_["path"]
    train_in_ = Path(path_in_) / path_split_["files"][0]

    # Cross-Validation utils
    models_stage = params_loader["models"]
    cross_val_stage = models_stage["cross_validation"]
    model_name = cross_val_stage["model"]
    cross_val_path_out_ = cross_val_stage["path"]

    # command-Line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--date", help="Recorded datetime during runtime execution."
    )
    parser.add_argument("-o", "--out", help="Output model directory")
    args = parser.parse_args()

    # date_time = date_time_record(args.date)
    cross_out_ = args.out
    os.makedirs(cross_out_, exist_ok=True)
    print(cross_out_)
    # Load training dataset
    dataframe = pd.read_csv(train_in_)
    x_all_ = dataframe[["text"]]
    y_all_ = dataframe["sentiment"]

    # Initiate cross validation process
    seed_ = models_stage["seed"]
    num_split_ = cross_val_stage["n_splits"]
    search_model_ = f"{model_name}_model.pkl"

    # Check if model file exist in development stage
    try:
        models = [
            str(f).rsplit("/", maxsplit=1)[-1]
            for f in list(Path(models_stage["dev"]["path"]).glob("*.pkl"))
        ]

        if not search_model_ in models:
            raise ValueError(
                f"`{search_model_}` model does not exists! Please run model_dev.py script."
            )

    except ValueError as e:
        sys.exit(f"{e}")

    model_pickle_ = Path(models_stage["dev"]["path"]) / search_model_
    clf = joblib.load(model_pickle_)

    scores_ = cross_valid_iteration(clf, x_all_, y_all_, num_split_, seed_)
    score_out_ = Path(cross_out_) / f"metrics.json"

    with open(score_out_, "w", encoding="utf-8") as f:
        json.dump(scores_, f, ensure_ascii=False, indent=4)

    plot_file_out_ = Path(cross_out_) / f"cv_scores_plot.png"
    create_model_comparison_plots(scores_, plot_file_out_)

    baseline_path = r"src/model_checkpoints/trained/metrics.json"
    validation_path = r"src/model_checkpoints/cross_valid/metrics.json"
    if hyperparameter_optimization_helper(baseline_path, validation_path): 
        print("Optimizing hyperparamters and retraining the model ...")
    else: 
        sys.exit("No need to optimize parameters.")

