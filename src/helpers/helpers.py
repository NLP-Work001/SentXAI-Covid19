import json
import os
from pathlib import Path
from typing import Union

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from mlflow.models import infer_signature
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


# >>>>> SECTION: MLFlow Experiments Logging
def default_pipeline_params(pipeline: BaseEstimator, model_name: str) -> dict:
    default_model_params = pipeline.named_steps[model_name].get_params()
    default_vect_params = (
        pipeline.named_steps["columntransformer"].transformers_[0][1].get_params()
    )

    pipeline_params = {"tfidf": default_vect_params, "model": default_model_params}
    return pipeline_params


def formated_metrics(metrics: dict) -> dict:
    metrics = {f"{k}-{i}": j for k, v in metrics.items() for i, j in v.items()}
    return metrics


class MLFlowExperiment:
    def __init__(self, path: str, path_split: str) -> None:
        # Create Experiment
        self.path = Path(path)
        self.path_split = path_split
        self.pipeline = joblib.load(f"{self.path}/sk_pipeline.pkl")
        self.model_name = list(self.pipeline.named_steps.keys())[-1]
        self.experiment_name = "SentXAI NLP Experiment - Sklearn"
        self.experiment_details = mlflow.get_experiment_by_name(self.experiment_name)

        if not self.experiment_details:
            # Experiment Description
            self.description = """Sentiment Analysis for Covid19 tweets dataset using scikit-learn algorithms.
                Various machine learning classifers are experimented."""

            self.experiment_tags = {
                "project-name": "SentXAI project",
                "mlflow.note.content": self.description,
            }

            mlflow.create_experiment(
                name=self.experiment_name, tags=self.experiment_tags
            )
            print("Experiment Created!")

        # Get the experiment ID
        self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id

    # Log Experiment Runs
    def track_model_experiment(self, x_df: pd.DataFrame) -> None:
        # Parameters formatting:
        with open(f"{self.path}/metrics.json", "r") as f:
            metrics = formated_metrics(json.load(f))

        # ToDo: Fix hard-coded path
        with open(f"{self.path_split}/data_info.json", "r", encoding="utf-8") as f:
            data_info = json.load(f)
        print("Started mlfow processs ...")
        with mlflow.start_run(
            experiment_id=self.experiment_id, run_name=self.model_name
        ) as run:
            # Log model parameters and metrics:
            params = {**default_pipeline_params(self.pipeline, self.model_name), **data_info}

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Log model artifacts:
            artifact_names = [
                Path(self.path) / c
                for c in os.listdir(self.path)
                if ".json" not in c and "pipeline" not in c
            ]
            for c in artifact_names:
                mlflow.log_artifact(c, artifact_path=self.model_name)

            print(f"Logging model <{self.model_name}> ...")
            model_signature = infer_signature(x_df, self.pipeline.predict_proba(x_df))
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                signature=model_signature,
                artifact_path=f"{self.model_name}/trained",
                registered_model_name=self.model_name,
            )
            print("Saving run information  ...")
            # Log model run_id to local directory:
            with open(f"{self.path}/experiment_info.json", "w", encoding="utf-8") as f:
                json.dump({"run_id": run.info.run_id, "experiment_id": self.experiment_id}, f)

            print("Logging experiment process completed!")


# <<<END SECTION>
