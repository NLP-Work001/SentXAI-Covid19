import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

from utils import get_output_label


# Plot ROC curve and ROC area
def plot_roc_auc_curve(
    model: BaseEstimator,
    encoding: LabelBinarizer,
    y_test_: np.matrix,
    x_test_: pd.DataFrame,
) -> None:
    """This function helps to plot roc and auc curve for model evaluation
    purposes.

    param: model -> scikit-learn type
    param: y_test_ -> matrix-like array (n_samples, n_classes)
    param: x_test_ -> pandas dataframe

    returns: None
    """
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(set(y_test_.argmax(axis=1)))
    y_scores = model.predict_proba(x_test_)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro average
    y_ravel = y_test_.ravel()
    score_ravel = y_scores.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_ravel, score_ravel)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    tpr["macro"] = mean_tpr
    fpr["macro"] = all_fpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Weighted average
    class_weights = np.sum(y_test_, axis=0) / np.sum(y_test_)
    weighted_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        weighted_tpr += class_weights[i] * np.interp(all_fpr, fpr[i], tpr[i])

    tpr["weighted"] = mean_tpr
    fpr["weighted"] = all_fpr
    roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

    # # Plot ROC curve
    plt.subplots(1, 1, figsize=(14, 6))

    for idx, w in zip(["*", "<", "1"], ["micro", "macro", "weighted"]):
        plt.plot(
            fpr[w],
            tpr[w],
            label=f"{w}-average ROC curve (area = {roc_auc[w]:0.2f})",
            marker=idx,
            linewidth=1.5,
        )

    for i in range(n_classes):
        label = get_output_label(encoding, i)
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"ROC curve of {label} (area = {roc_auc[i]:0.2f}",
            linewidth=1.5,
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Evaluation")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # Save plot
    plot_roc_auc_curve(model, y_test_, x_test_)
