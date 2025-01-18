import json
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline


class ModelEvaluator:
    """Evaluates model performance and suggests optimization actions."""

    def __init__(self, baseline_results: dict, cv_results: dict):
        self.baseline_results = baseline_results
        self.baseline_score = np.mean(list(baseline_results["test"].values()))
        self.cv_results = cv_results

    def __calculate_average_score(self, fold: str = "train") -> float:
        return np.mean(list(self.cv_results[fold].values()))

    def make_decision(self) -> tuple[str, bool, float]:
        train_score = self.__calculate_average_score("train")
        val_score = self.__calculate_average_score("valid")
        score_diff = np.round(train_score - val_score, 3)

        if self.baseline_score >= 0.6:
            if score_diff >= 0.1:
                return "Model is likely overfitting.", True, score_diff
            if 0 <= score_diff <= 0.2:
                return "Passed", False, score_diff
        return "EXIT: Model is likely underfitting", False, score_diff


# Optimization decision helper
def hyperparameter_optimization_helper(
    baseline_path: str, validation_path: str
) -> bool:
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_scores = json.load(f)

    with open(validation_path, "r", encoding="utf-8") as f:
        validation_scores = json.load(f)

    decision = ModelEvaluator(baseline_scores, validation_scores)
    message, bool_output, error_score = decision.make_decision()
    print("MESSAGE: ", message)
    print("SCORE_ERROR: ", error_score)
    return bool_output


def create_model_comparison_plots(model_results: dict, path_out: str) -> None:
    """
    Creates bar plots to compare performance metrics across different models."""

    metrics = [v for v in model_results.values()]
    columns = next(iter([list(c.keys()) for c in model_results.values()]))
    index = [k for k in model_results.keys()]
    scores = pd.DataFrame(metrics, index=index, columns=columns)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, c in enumerate(columns):
        sns.barplot(scores, x=scores.index, y=c, hue=scores.index, ax=axes[i], gap=0.5)
        axes[i].set_title(f"`{c}` scores comparison", alpha=0.7, fontweight=10)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_xlabel("Fold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(path_out, dpi=300)
    # plt.show()


# Loads configuration yml file
def config_loader(file_name: str) -> dict:
    with open(file_name, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


# TFIDf token function
def word_processor(doc: str):
    return doc


# Model training pipeline
def model_pipeline(
    baseline: BaseEstimator, vect_ngram_: Union[tuple, list] = (1, 1), vect_norm_="l2"
):
    if isinstance(vect_ngram_, list) and vect_norm_ is None:
        vect = TfidfVectorizer(preprocessor=word_processor)
    else:
        vect = TfidfVectorizer(
            ngram_range=vect_ngram_, norm=vect_norm_, preprocessor=word_processor
        )

    ct = make_column_transformer((vect, "text"), remainder="drop")
    pipeline = make_pipeline(ct, baseline)
    return pipeline


# Helps to dynamically update configuration file parameters
def update_configs_yml(file_path: str, key: str, new_value: bool) -> None:
    
    configs = config_loader(file_path)
    get_retrain = configs["models"]["train"]
    
    if key in get_retrain:
        get_retrain["optimize"] = new_value
    else:
        print(f"Key `{key}` not found in the YML file.")
        return
    
    with open("configs.yml", "w") as f:
        yaml.dump(configs, f, default_flow_style=False)
        print(f"Updated `{key}` to `{new_value}` in {file_path}.")
