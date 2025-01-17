import sys
from typing import Union
import numpy as np

def model_criteria(
    error_score: float, baseline_score: float
) -> Union[str, bool]:
    """ """

    if baseline_score >= 0.6:
        if error_score >= 0.15:
            return "Model is likely overfitting.", True
        if 0 <= error_score <= 0.2:
            return "passed", False
        else:
            sys.exit("EXIT: Model Failed! It is likely underfitting.")
    else:
        sys.exit("EXIT: Model score does not meet intial selection criteria.")

if __name__ == "__main__":
    training_metrics = {
        "train": {
            "f1": 0.89,
            "roc_auc": 0.99
        },
        "test": {
            "f1": 0.56,
            "roc_auc": 0.75
        }
    }
    
    validation_scores = {
        "time_lapsed": "0:00:24",
        "train_f1": 0.89,
        "train_roc_auc": 0.98,
        "val_f1": 0.6,
        "val_roc_auc": 0.79
    }
        
    scores = list(training_metrics["test"].values())
    
    error = validation_scores["train_roc_auc"] - validation_scores["val_roc_auc"]
    print(scores)
    print("Average score: ", np.mean(scores))
    print("Error: ", error)
    print(model_criteria(error, np.mean(scores)))