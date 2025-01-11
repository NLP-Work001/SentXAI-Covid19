import os
from pathlib import Path
from typing import Union

import yaml
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB


# TFIDf token function
def word_processor(doc: str):
    return doc


# Calculate model metrics and output as a dictionary of those metrics
def calculate_metric_score(
    model: BaseEstimator, x_values: pd.DataFrame, y_true: list
) -> dict:

    f1 = f1_score(y_true, model.predict(x_values), average="weighted")
    roc_auc = roc_auc_score(
        y_true, model.predict_proba(x_values), average="weighted", multi_class="ovo"
    )

    metrics = {"f1": np.round(f1, 2), "roc_auc": np.round(roc_auc, 2)}
    return metrics


def cross_valid_mean_score(metrics: list, alias="train") -> dict:

    keys = list(next(iter(metrics)))
    values = [list(score.values()) for score in metrics]
    score_avg = np.mean(values, axis=0)
    metric_scores = {f"{alias}_{n}": np.round(v, 2) for n, v in zip(keys, score_avg)}

    return metric_scores


# Load train/test split data
def load_train_split(file_name: str) -> pd.DataFrame:

    # ToDo: Modified the path code
    folder_dir = Path(os.path.abspath(".")) / "data/split"
    file_path = os.path.join(folder_dir, file_name)
    df = pd.read_csv(file_path)
    return df


# Convert numerical labels into named labels
def get_output_label(encoding, index, size=3) -> str:
    try:
        if index >= size:
            raise ValueError("Index must be less than number of classes.")

        arr = np.zeros(3, dtype=int)
        arr[index] = 1
        arr = arr.reshape(1, size)

        output = encoding.inverse_transform(arr)[0]
        return output
    except ValueError as e:
        print(f"Error: {e}")
        return ""


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


# Select model for cross-val or fine-tuning
def select_model(model: str, seed=0) -> BaseEstimator:
    models = {
        "bayes": MultinomialNB(),
        "svc": SVC(probability=True),
        "logistic": LogisticRegression(random_state=seed),
        "rforest": RandomForestClassifier(random_state=seed),
        "tree": DecisionTreeClassifier(random_state=seed),
    }
    return models[model]


# Read Script parameters
def load_parameters(file_name: str) -> dict:
    with open(file_name, "r") as f:
        params = yaml.safe_load(f)
    return params


# date & time recorder
def date_time_record(date: str) -> str:
    str_list = date.split()
    execution_time = []

    for idx, j in enumerate(["-", ":"]):
        execution_time.append("".join(str_list[idx].split(j)))

    return "_".join(execution_time)
