import sys
from typing import Union
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

    def _make_decision(self) -> tuple[str, bool]:
        train_score = self._calculate_average_score("train")
        val_score = self._calculate_average_score("val")
        score_diff = train_score - val_score

        if self.baseline_score >= 0.6:
            if score_diff >= 0.15:
                return "Model is likely overfitting.", True
            elif 0 <= score_diff <= 0.2:
                return "Passed", False
            else:
                return "Model is likely underfitting", False
        else:
            return "EXIT: Model score does not meet initial selection criteria.", False

def create_model_comparison_plots(model_results: dict) -> None:
    """
    Creates bar plots to compare performance metrics across different models."""

    metrics = [v for v in model_results.values()]
    columns =  next(iter([list(c.keys()) for c in model_results.values()]))
    index =  [k for k in model_results.keys()]
    scores = pd.DataFrame(metrics, index=index, columns=columns)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    for i, c in enumerate(columns):
        plot = sns.barplot(scores, x=scores.index, y=c, hue=scores.index, ax=axes[i], gap=0.5)
        axes[i].set_title(f"`{c}` scores comparison", alpha=0.7, fontweight=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlabel("Fold")

    plt.tight_layout()
    plt.show()


# USAGE:

# create_model_comparison_plots(validation_scores)

# decision = DecisionML(training_metrics, validation_scores)
# message, bool_output = decision._decision_fn()

# if bool_output:
#     print("call hyper-tuning function")
# else:
#     sys.exit(message)



    