import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    multilabel_confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import LabelBinarizer

sys.path.append(str("src/helpers"))

from utils import (
    calculate_model_metrics,
    date_time_record,
    get_output_label,
    model_pipeline,
)

from helpers import config_loader, create_model_comparison_plots


# Plot confusion matrix for multiclasses
def plot_multilabel_cm(
    model: BaseEstimator,
    encoding: LabelBinarizer,
    y_true_: np.matrix,
    x_test_: pd.DataFrame,
    out_: str,
) -> None:
    # Compute multilabel confusion matrix
    y_scores = model.predict(x_test_)
    mcm = multilabel_confusion_matrix(y_true_.argmax(axis=1), y_scores)

    # Plot each confusion matrix
    fig, axes = plt.subplots(1, len(mcm), figsize=(15, 5))
    for i, matrix in enumerate(mcm):
        label = get_output_label(encoding, i)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=[f"Not {label}", label]
        )
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

    model_name = list(model.named_steps.values())[-1].__class__.__name__.lower()

    plt.title(f"Roc Curve for {model_name}.")
    plt.legend(loc="lower right")

    fig.savefig(Path(out_) / "roc_auc_curve.png", dpi=dpi)


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
    params_["model"] = data["model"]["params"]

    return params_


def _training(
    baseline_model: BaseEstimator,
    train_file_path_: str,
    test_file_path_: str,
    out_: dict,
) -> None:
    # Load training & testing dataset

    _train = pd.read_csv(train_file_path_)
    x_train_ = _train[["text"]]
    y_train_ = _train["sentiment"]

    _test = pd.read_csv(test_file_path_)
    x_test_ = _test[["text"]]
    y_test_ = _test["sentiment"]

    # Label Encoding
    binarizer = LabelBinarizer(sparse_output=False)

    # label_encoding = LabelEncoder()
    y_train_ = binarizer.fit_transform(y_train_)
    y_test_ = binarizer.transform(y_test_)

    # training model
    joblib.dump(binarizer, out_["encoder"])
    model = None
    if out_["retrain"]:       
        model = model_pipeline(baseline_model, out_["ngram_range"],  out_["norm"]).fit(x_train_, y_train_.argmax(axis=1))
    else:
        model = model_pipeline(baseline_model).fit(x_train_, y_train_.argmax(axis=1))

    train_scores = calculate_model_metrics(model, x_train_, y_train_.argmax(axis=1))
    test_scores = calculate_model_metrics(model, x_test_, y_test_.argmax(axis=1))

    # Log model artifacts
    scores = {"train": train_scores, "test": test_scores}

    with open(out_["metric"], "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    joblib.dump(model, out_["model"])
    labels_ = np.unique(binarizer.inverse_transform(y_test_))
    plot_roc_auc_curve(model, binarizer, y_test_, x_test_, out_["plot_out"])
    plot_multilabel_cm(model, binarizer, y_test_, x_test_, out_["plot_out"])
    create_model_comparison_plots(scores, Path(out_["plot_out"]) / "metrics_plots.png")

    # Log classification report
    y_pred = model.predict(x_test_)
    report = classification_report(
        y_test_.argmax(axis=1), y_pred, target_names=labels_, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(Path(out_["path"]) / "classification_report.csv", index=True)


def main() -> None:
    print("Started training ...")

    # ToDo: WriteUp retraing logic before connecting MLflow and DagsHUB
    # ToDo: Add Jenkins and Github Actions operations
    training_in_ = sys.argv[1]
    training_out_ = sys.argv[2]
    os.makedirs(training_out_, exist_ok=True)

    # Load configs file
    params_loader = config_loader("configs.yml")

    # Reading data file
    files_path_ = params_loader["data"]["split"]
    train_file_path_ = Path(training_in_) / files_path_["files"][0]
    test_file_path_ = Path(training_in_) / files_path_["files"][1]

    # Train utils
    trained_param = params_loader["models"]["train"]
    model_name_ = trained_param["model"]
    encoder_file_ = trained_param["encoder"]
    metrics_file_ = trained_param["metrics"]

    # Setup output files
    baseline_path_ = (
        Path(params_loader["models"]["path"])
        / params_loader["models"]["baseline"]["path"]
    )
    search_model_ = f"{model_name_}_model.pkl"

    # Check if model file exist in development stage
    try:
        models = [
            str(f).split("/")[-1] for f in list(Path(baseline_path_).glob("*.pkl"))
        ]

        if not search_model_ in models:
            raise ValueError(
                f"`{search_model_}` model does not exists! Please run model_dev.py script."
            )

    except ValueError as e:
        sys.exit(f"{e}")

    # Setup Files output
    encoder_file_out_ = Path(training_out_) / encoder_file_
    metrics_file_out_ = Path(training_out_) / metrics_file_
    output_model_file_ = Path(training_out_) / "model.pkl"

    out_ = {
        "path": training_out_,
        "model": output_model_file_,
        "metric": metrics_file_out_,
        "encoder": encoder_file_out_,
        "plot_out": Path(training_out_),
        "retrain": False
    }

    # Model training
    model = joblib.load(Path(baseline_path_) / search_model_)

    # Get optimized parameters
    best_params = _tune_params_loader("src/model_checkpoints/tuned")
    model_params_ = best_params["model"]
    vect_params_ = best_params["vectorizer"]

    print("Show ouput: ", trained_param["optimize"])
    if trained_param["optimize"]:
        print("Model retraining ...")
        out_["ngram_range"] = vect_params_["ngram_range"]
        out_["norm"] = vect_params_["norm"]
        out_["retrain"] = True
        model.set_params(**model_params_)
        
    _training(model, train_file_path_, test_file_path_, out_)


if __name__ == "__main__":
    main()
