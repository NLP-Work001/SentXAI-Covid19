import json
import os
from argparse import ArgumentParser
import pandas as pd
import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path
import numpy as np
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import sys
import matplotlib.pyplot as plt

from utils import (
    calculate_metric_score,
    load_parameters,
    get_output_label,
    model_pipeline,
    date_time_record
)

# Plot confusion matrix for multiclasses
def plot_multilabel_cm(
    model: BaseEstimator,
    encoding: LabelBinarizer,
    y_true_: np.matrix,
    x_test_: pd.DataFrame,
    out_: str
) -> None:

    # Compute multilabel confusion matrix
    y_scores = model.predict(x_test_)
    mcm = multilabel_confusion_matrix(y_true_.argmax(axis=1), y_scores)

    # Plot each confusion matrix
    fig, axes = plt.subplots(1, len(mcm), figsize=(15, 5))
    for i, matrix in enumerate(mcm):
        label = get_output_label(encoding, i)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[f"Not {label}", label])
        disp.plot(ax=axes[i], cmap=plt.cm.Blues, colorbar=False)
        axes[i].set_title(f"Confusion matrix for '{label}'")
        axes[i].grid(False)

    plt.tight_layout()

    path_out = Path(out_) / "confusion_matric_plot.png"
    fig.savefig(path_out, dpi=250)

# Plot ROC curve and ROC area
def plot_roc_auc_curve(
    model: BaseEstimator,
    encoding: LabelBinarizer,
    y_true_: np.matrix,
    x_test_: pd.DataFrame,
    out_: str,
    dpi=100,
) -> None:
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(set(y_true_.argmax(axis=1)))
    y_scores = model.predict_proba(x_test_)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_[:, i], y_scores[:, i])
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

    fig.savefig(Path(out_) / f"roc_auc_curve.png", dpi=dpi)

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

    model = model_pipeline(baseline_model).fit(
        x_train_, y_train_.argmax(axis=1)
    )

    train_scores = calculate_metric_score(model, x_train_, y_train_.argmax(axis=1))
    test_scores = calculate_metric_score(model, x_test_, y_test_.argmax(axis=1))

    # Log model artifacts
    scores = {"train": train_scores, "test": test_scores}

    with open(out_["metric"], "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    joblib.dump(model, out_["model"])
    labels_ = np.unique(binarizer.inverse_transform(y_test_))
    plot_roc_auc_curve(model, binarizer, y_test_, x_test_, out_["plot_out"])
    plot_multilabel_cm(model, binarizer, y_test_, x_test_, out_["plot_out"])
    
    # Log classification report
    y_pred = model.predict(x_test_)
    report = classification_report(y_test_.argmax(axis=1), y_pred, target_names=labels_ ,  output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(Path(out_["path"]) / "classification_report.csv", index=True)

    print("Training Scores: ", train_scores)
    print("Testing Scores: ", test_scores)


def main() -> None:
    print("Started training ...")



    # command-Line arguments
    parser = ArgumentParser()
    parser.add_argument("-d", "--date", help="Recorded date during runtime execution.")
    parser.add_argument("-o", "--out", help="Output model directory")
    args = parser.parse_args()

    date_time = date_time_record(args.date)
    model_output_folder_ = args.out
    os.makedirs(model_output_folder_, exist_ok=True)
    # Load configs file
    params_loader = load_parameters("config.yml")

    # Reading data file
    parent_split_ = params_loader["data"]
    path_split_ = parent_split_["split"]
    path_in_ = path_split_["path"]
    train_in_ = Path(path_in_) / path_split_["files"][0]
    test_in_ = Path(path_in_) / path_split_["files"][1]

    # Train utils
    train_dev_ = params_loader["models"]
    # train_path_ = train_dev_["train"]["path"]
    model_name_ = train_dev_["train"]["model"]
    encoder_file_ = train_dev_["train"]["encoder"]
    metrics_file_ = train_dev_["train"]["metrics"]

    # Setup output files
    model_out_ = train_dev_["dev"]["path"]
    search_model_ = f"{model_name_}_model.pkl"
   
   # Check if model file exist in development stage
    try:
        models = [str(f).split("/")[-1] for f in list(Path(model_out_).glob("*.pkl"))]
        # print(models)
        if not search_model_ in models:
            raise ValueError(f"`{search_model_}` model does not exists! Please run model_dev.py script.")

    except ValueError as e:
        sys.exit(f"{e}")

    # Setup Files output
    encoder_file_out_ = Path(model_output_folder_) / encoder_file_
    metrics_file_out_ = Path(model_output_folder_) / metrics_file_
    output_model_file_ =  Path(model_output_folder_) / "model.pkl"

    out_ = {
        "path": model_output_folder_,
        "model":  output_model_file_,
        "metric": metrics_file_out_,
        "encoder": encoder_file_out_,
        "plot_out":  Path(model_output_folder_),
    }

    # Model training
    model = joblib.load(Path(model_out_) / search_model_)
    # model.set_params(**optimized_params["model"])
    _training(model, train_in_, test_in_, out_)


if __name__ == "__main__":
    main()
