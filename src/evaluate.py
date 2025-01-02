import numpy as np
from .utils import *
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve, auc

# Plot ROC curve and ROC area
def plot_roc_auc_curve(model: BaseEstimator, y_test_, X_test_) -> None:
    
    fpr = dict()
    tpr = dict() 
    roc_auc = dict()
    n_classes = len(set(y_test_.argmax(axis=1)))
    y_scores = model.predict_proba(X_test_)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])



    # Micro average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_.ravel(), y_scores.ravel())
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
        plt.plot(fpr[w], tpr[w], label=f'{w}-average ROC curve (area = {roc_auc[w]:0.2f})', marker=idx, linewidth=1.5)
        
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of {get_output_label(i)} (area = {roc_auc[i]:0.2f}', linewidth=1.5)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Evaluation')
    plt.legend(loc="lower right")
    plt.show()
    
if __name__ == "__main__":
    # Save plot
    plot_roc_auc_curve(model, y_test_, X_test_)