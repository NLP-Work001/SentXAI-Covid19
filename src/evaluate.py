from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

from utils import get_output_label, load_parameters, load_train_split


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
    plt.title("ROC-AUC Evaluation")
    plt.legend(loc="lower right")
    fig.savefig(Path(out_) / "roc_auc_curve.png", dpi=dpi)


if __name__ == "__main__":
    param_loader = load_parameters("params.yml")
    training_arg = param_loader["training"]
    model_name = training_arg["model"]

    eval_out_ = Path(training_arg["path_out"]) / model_name

    pickle_file_ = Path(eval_out_) / "model.pkl"
    model = joblib.load(pickle_file_)

    encoder_file_ = Path(eval_out_) / "encoder_binarizer.pkl"
    encoding = joblib.load(encoder_file_)

    # ToDo: Remove head(1000) for full data testing
    _test = load_train_split("test.csv").head(1000)
    x_test_ = _test[["text"]]
    y_test_ = _test["sentiment"]
    y_test_ = encoding.transform(y_test_)

    plot_roc_auc_curve(model, encoding, y_test_, x_test_, eval_out_)
