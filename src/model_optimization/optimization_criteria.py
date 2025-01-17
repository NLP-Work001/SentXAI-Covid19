import numpy as np

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


if __name__ == "__main__":
  # Usage 
    training_metrics = {
        "train": {
            "f1": 0.89,
            "roc_auc": 0.99
        },
        "test": {
            "f1": 0.53,
            "roc_auc": 0.75
        }
    }

    validation_scores = {
        "time_lapsed": "0:00:24",
        "train": {
            "f1": 0.89,
            "roc_auc": 0.98
        },
        "val": {
            "f1": 0.6,
            "roc_auc": 0.79
        }
    }

    decision = DecisionML(training_metrics, validation_scores)
    message, bool_output = decision._decision_fn()

    if bool_output:
        print("call hyper-tuning function")
    else:
        sys.exit(message)