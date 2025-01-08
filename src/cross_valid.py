import os
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from utils import (
    calculate_average_cv,
    calculate_metric_score,
    load_parameters,
    model_pipeline,
)


# Barplots for metric comparisons
def cross_valid_score_plot(scores: dict, out_path: str, img_pixel=100) -> None:
    """Helps to plot bar-graphs to visualize whether a model
    is overfitting or underfitting. The y-axis represent the
    metric scores and the x-axis represents respective selected
    metric name.

    params: scores in dictionary form.
    returns: None
    """
    filter_metric = [
        (
            (c.split("_")[0], "_".join(c.split("_")[1:]), k)
            if len(c.split("_")) > 2
            else (*c.split("_"), k)
        )
        for c, k in scores.items()
        if "time" not in c
    ]

    cols = ["cv_fold", "metric", "score"]
    metric_df = pd.DataFrame(filter_metric, columns=cols).sort_values(
        "metric", ascending=False
    )

    # Seaborn plot
    fig, ax = plt.subplots(figsize=(10, 4))
    # plt.style.use("ggplot")
    sns.set_theme()

    sns.barplot(
        metric_df,
        x="metric",
        y="score",
        ax=ax,
        width=0.5,
        hue="cv_fold",
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

    print(f"Image file saved into {out_path}")


def cross_validation_func(
    baseline: BaseEstimator, x: pd.DataFrame, y: pd.DataFrame
) -> dict:
    """Helps to easily perform model cross validation.

    param: baseline -> selected model to validate
    param: x -> pandas dataframe
    param: y -> pandas dataframe

    returns: both training and validation score metrics
    """
    pipe = model_pipeline(baseline)

    label_encoding = LabelEncoder()
    y_all = label_encoding.fit_transform(y)

    # Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

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
    training_score = calculate_average_cv(train_scores)
    validation_score = calculate_average_cv(val_scores, "val")
    metric_scores = {"time_lapsed": total_time, **training_score, **validation_score}

    return metric_scores


def _date_time_record(date: str) -> str:
    str_list = date.split()
    execution_time = []

    for idx, j in enumerate(["-", ":"]):
        execution_time.append("".join(str_list[idx].split(j)))

    return "_".join(execution_time)


if __name__ == "__main__":
    # Retrieve file path
    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--split", help="accepts folder path from train/test split."
    )
    parser.add_argument(
        "-d", "--date", help="retrieves datetime during script execution."
    )
    parser.add_argument("-o", "--out", help="contains model output path.")

    args = parser.parse_args()

    # Load training dataset for model training
    parameters = load_parameters("params.yml")
    train_input_file = parameters["data"]["split"][0]
    file_path = Path(args.split) / train_input_file

    date_time = _date_time_record(args.date)
    file_out_name = "cross_valid" + "_" + date_time + ".png"

    data = pd.read_csv(file_path)
    x_all_ = data[["text"]]
    y_all_ = data[["sentiment"]]

    # Training: cross validation process
    seed = 43
    models = {
        "bayes": MultinomialNB(),
        "svm": SVC(probability=True),
        "lr": LogisticRegression(max_iter=100, random_state=seed),
        "rf": RandomForestClassifier(random_state=seed),
        "tree": DecisionTreeClassifier(random_state=seed),
    }

    # ToDo: check how roc_auc score are calculated for ensemble models
    model_type = parameters["cross_valid"]["model"]
    if model_type == "vote":
        model = VotingClassifier(list(models.items()), voting="hard")
    else:
        model = models[model_type]

    print("Started cross validation ...")
    model_name = f"{model.__class__.__name__}".lower()
    scores_ = cross_validation_func(model, x_all_, y_all_)

    plot_output = Path(f"{args.out}/{model_name}") / file_out_name
    os.makedirs(plot_output.parent, exist_ok=True)

    cross_valid_score_plot(scores_, plot_output)
    model_out = Path(plot_output.parent) / "model.pkl"
    joblib.dump(model, model_out)
    print("Process ended.")
    print("Parent path: ", plot_output.parent)
    print("File input path: ", file_path)
    print("Image file output name: ", file_out_name)
    print("Plot output file: ", plot_output)
