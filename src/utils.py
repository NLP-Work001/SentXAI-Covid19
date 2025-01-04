import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from datetime import timedelta
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score


# TFIDf token function
def word_processor(doc: str):
    return doc


# Calculate model metrics and output as a dictionary of those metrics
def calculate_model_metrics(model: BaseEstimator, X_values: pd.DataFrame, y_true: list) -> dict:

    f1 = f1_score(y_true, model.predict(X_values), average="weighted")
    roc_auc =  roc_auc_score(y_true, model.predict_proba(X_values),  average="weighted", multi_class="ovo")

    metrics = dict(f1=np.round(f1, 2), roc_auc=np.round(roc_auc, 2))
    return metrics


def calculate_average_cv(metrics:list, alias="train") -> dict:
    """Calculates the average cross-validation scores.
    params: metrics - list of cv metrics
    returns: average cv scores
    """
    keys = list(next(iter(metrics)))
    values = [list(score.values()) for score in metrics]
    score_avg = np.mean(values, axis=0)
    cv_metrics = {f"{alias}_{n}": np.round(v, 2) for n, v in zip(keys, score_avg)}
    cv_scores = {f"{alias}_avg": np.round(np.mean(score_avg), 2), **cv_metrics}
    
    return cv_scores 


# Load train/test split data
def load_train_split(file_name: str) -> pd.DataFrame:
    """ Load train/test dataset from splits folder.
    params: File path
    returns: pandas dataframe
    """
    folder_dir = Path(os.path.abspath("..")) / "data/splits"
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