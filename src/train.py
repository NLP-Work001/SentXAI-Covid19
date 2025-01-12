import json
import os
from argparse import ArgumentParser
import pandas as pd
import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve

from utils import (
    calculate_metric_score,
    load_parameters,
    get_output_label,
    model_pipeline,
)


# Plot ROC curve and ROC area
def plot_roc_auc_curve(
    model: BaseEstimator,
    encoding: LabelBinarizer,
    y_test_: np.matrix,
    x_test_: pd.DataFrame,
    out_: str,
    dpi=100,
) -> None:
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(set(y_test_.argmax(axis=1)))
    y_scores = model.predict_proba(x_test_)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # # Plot ROC curve
    plt.style.use("ggplot")
    fig, _ = plt.subplots(1, 1, figsize=(14, 6))

    for i in range(n_classes):
        label = get_output_label(encoding, i)
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"ROC curve of {label}, (AUC = {roc_auc[i]:0.2f}",
            linewidth=1.5,
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    model_name = (
        list(model.named_steps.values())[-1]
        .__class__.__name__.lower()
    )

    plt.title(f"Roc Curve for {model_name}.")
    plt.legend(loc="lower right")

    fig.savefig(out_, dpi=dpi)

# Access optimized model parameters
def _tune_params_loader(path: str) -> dict:
    params_ = {}
    file_in_ = Path(path) / "best_params.json"

    with open(file_in_, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, j in data.items():
        params_[i] = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in j["params"].items()
        }

    return params_


def _training(baseline_model: BaseEstimator, train_in_: str, test_in_: str, out_: dict) -> None:
    # Load training & testing dataset

    _train = pd.read_csv(train_in_)
    x_train_ = _train[["text"]]
    y_train_ = _train["sentiment"]

    _test = pd.read_csv(test_in_)
    x_test_ = _test[["text"]]
    y_test_ = _test["sentiment"]

    # Label Encoding
    binarizer = LabelBinarizer(sparse_output=False)

    # label_encoding = LabelEncoder()
    y_train_ = binarizer.fit_transform(y_train_)
    y_test_ = binarizer.transform(y_test_)

    # training model
    joblib.dump(binarizer, out_["encoder"])

    norm = out_["vectorizer"]["norm"]
    ngram_range = out_["vectorizer"]["ngram_range"]

    model = model_pipeline(baseline_model, ngram_range, norm).fit(
        x_train_, y_train_.argmax(axis=1)
    )

    train_scores = calculate_metric_score(model, x_train_, y_train_.argmax(axis=1))
    test_scores = calculate_metric_score(model, x_test_, y_test_.argmax(axis=1))

    scores = {"train": train_scores, "test": test_scores}

    with open(out_["metric"], "w", encoding="utf-8") as f:
        json.dump(scores, f)

    joblib.dump(model, out_["model"])
    plot_roc_auc_curve(model, binarizer, y_test_, x_test_, out_["plot_out"])

    print("Training Scores: ", train_scores)
    print("Testing Scores: ", test_scores)

def main() -> None:
    print("Started training ...")
    params_loader = load_parameters("params.yml")

    # Reading data file
    parent_ = params_loader["data"]
    path_in_ = parent_["split"]["path"]
    train_in_ = Path(path_in_) / parent_["split"]["file"][0]
    test_in_ = Path(path_in_) / parent_["split"]["file"][1]


    # Train utils
    dev = params_loader["dev"]
    model_in_ = dev["path"]
    file_in_ = dev["file"]
    train_path_ = dev["train"]["path"]
    model_type = dev["train"]["model"]

    fetch_model_pickle_ = Path(f'{model_in_}/{dev["cross-valid"]["path"]}/{model_type}') / file_in_
    fetch_param_in_ = Path(f'{model_in_}/{dev["fine-tune"]["path"]}/{model_type}')
    path_out_ = Path(f"{model_in_}/{train_path_}/{model_type}")
    os.makedirs(path_out_, exist_ok=True)

    optimized_params = _tune_params_loader(fetch_param_in_)

    vectorizer_params = optimized_params["vectorizer"]

    out_ = {
        "model":  Path(path_out_) / file_in_,
        "metric": Path(path_out_) / dev["train"]["metric"],
        "encoder": Path(path_out_) / dev["train"]["encoder"],
        "plot_out":  Path(path_out_) / "roc_auc_curve.png",
        "vectorizer": vectorizer_params,
    }



    # Model training
    model = joblib.load(fetch_model_pickle_)
    model.set_params(**optimized_params["model"])

    _training(model, train_in_, test_in_, out_)


if __name__ == "__main__":
    main()
