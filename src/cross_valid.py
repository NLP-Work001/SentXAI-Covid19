import os
import time
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
from tqdm import tqdm

from utils import (
    cross_valid_mean_score,
    calculate_metric_score,
    load_parameters, model_pipeline,
    date_time_record,
    select_model
)


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


def cross_valid_iteration(
    baseline: BaseEstimator, x: pd.DataFrame, y: pd.DataFrame,
    num_split: int, seed: float
) -> dict:

    pipe = model_pipeline(baseline)

    label_encoding = LabelEncoder()
    y_all = label_encoding.fit_transform(y)

    # Cross-Validation
    cv = StratifiedKFold(n_splits=num_split, shuffle=True, random_state=seed)

    train_scores = []
    val_scores = []
    time_start = time.time()
    time_lapsed = None

    for idx, (train_idx, val_idx) in tqdm(enumerate(cv.split(x, y_all))):
        print(" Iteration Count:", idx + 1)
        x_train_, x_val_ = x.iloc[train_idx], x.iloc[val_idx]
        y_train_, y_val_ = y_all[train_idx], y_all[val_idx]

        pipe.fit(x_train_, y_train_)

        train_score = calculate_metric_score(pipe, x_train_, y_train_)
        val_score = calculate_metric_score(pipe, x_val_, y_val_)

        train_scores.append(train_score)
        val_scores.append(val_score)
        time_lapsed = time.time() - time_start

    # Training time-lapsed
    seconds = np.round(time_lapsed)
    total_time = str(timedelta(seconds=seconds))

    # Cross-validated metric scores
    training_score = cross_valid_mean_score(train_scores)
    validation_score = cross_valid_mean_score(val_scores, "val")
    metric_scores = {"time_lapsed": total_time, **training_score, **validation_score}

    return metric_scores


if __name__ == "__main__":
    print("Started cross validation ...")
    params_loader = load_parameters("params.yml")

    # Reading data file
    parent_ = params_loader["data"]
    path_in_ = parent_["split"]["path"]

    train_in_ = Path(path_in_) / parent_["split"]["file"][0]
    
    # Cross-Validation utils
    dev = params_loader["dev"]
    
    dev_path = dev["path"]
    cross_valided = dev["cross-valid"]
    cv_path_ = cross_valided["path"]
    cv_model = cross_valided["model"]
    cv_path_out_ = Path(f"{dev_path}/{cv_path_}/{cv_model}")
    os.makedirs(cv_path_out_, exist_ok=True)
    
    # command-Line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--date", help="Recorded date during runtime execution."
    )
    
    args = parser.parse_args()
    
    date_time = date_time_record(args.date)
    file_out_ = f"cross_valid_{date_time}.png"

    # Load training dataset
    dataframe = pd.read_csv(train_in_)
    x_all_ = dataframe[["text"]]
    y_all_ = dataframe["sentiment"]

    # Initiate cross validation process
    pickle_ = dev["file"]
    num_split_ = cross_valided["n_split"]
    seed_ = parent_["split"]["seed"]
    
    clf = select_model(cv_model, seed_)
    scores_ = cross_valid_iteration(clf, x_all_, y_all_, num_split_, seed_)
    
    score_out_ = Path(cv_path_out_) / f"cv_scores_{date_time}.json"
    
    with open(score_out_, "w", encoding="utf-8") as f:
        json.dump(scores_, f, ensure_ascii=False, indent=4)
        
    plot_file_out_ = Path(cv_path_out_) / file_out_
    plot_cross_valid_score(scores_, plot_file_out_)
    
    model_file_out_ = Path(cv_path_out_) / pickle_
    joblib.dump(clf, model_file_out_)
    
    # print(seed_)
    # print(train_in_)
    # print(cv_path_out_)
    # print(clf.__class__.__name__)
    # print(plot_file_out_)
    # print(model_file_out_)
    
