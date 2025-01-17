import sys
import json
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Evaluates model performance and suggests optimization actions."""

    def __init__(self, baseline_results: dict, cv_results: dict):
        self.baseline_results = baseline_results
        self.baseline_score = np.mean(list(baseline_results["test"].values()))
        self.cv_results = cv_results

    def _calculate_average_score(self, fold: str = "train") -> float:
        return np.mean(list(self.cv_results[fold].values()))

    def _make_decision(self) -> tuple[str, bool, float]:
        train_score = self._calculate_average_score("train")
        val_score = self._calculate_average_score("valid")
        score_diff = np.round(train_score - val_score, 3)

        if self.baseline_score >= 0.6:
            if score_diff >= 0.1:
                return "Model is likely overfitting.", True, score_diff
            elif 0 <= score_diff <= 0.2:
                return "Passed", False, score_diff
            else:
                return "Model is likely underfitting", False, score_diff
        else:
            return "EXIT: Model score does not meet initial selection criteria.", False, score_diff

# Optimization decision helper
def hyperparameter_optimization_helper(baseline_path: str, validation_path: str) -> bool:

    with open(baseline_path, "r") as f1, open(validation_path, "r") as f2:
        baseline_scores = json.load(f1)
        validation_scores = json.load(f2)

    decision = ModelEvaluator(baseline_scores, validation_scores)
    message, bool_output, error_score = decision._make_decision()
    print("MESSAGE: ", message)
    print("SCORE_ERROR: ", error_score)
    return bool_output

def create_model_comparison_plots(model_results: dict, path_out: str) -> None:
    """
    Creates bar plots to compare performance metrics across different models."""

    metrics = [v for v in model_results.values()]
    columns =  next(iter([list(c.keys()) for c in model_results.values()]))
    index =  [k for k in model_results.keys()]
    scores = pd.DataFrame(metrics, index=index, columns=columns)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, c in enumerate(columns):
        plot = sns.barplot(scores, x=scores.index, y=c, hue=scores.index, ax=axes[i], gap=0.5)
        axes[i].set_title(f"`{c}` scores comparison", alpha=0.7, fontweight=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlabel("Fold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(path_out, dpi=300)
    # plt.show()


    