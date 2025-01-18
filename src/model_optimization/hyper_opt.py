import os
import subprocess
import sys

sys.path.append(str("src/helpers"))
sys.path.append(str("src/model_optimization"))

from argparse import ArgumentParser
from pathlib import Path

import joblib
import pandas as pd
from parameter_configs import model_params_, vector_params_
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from utils import date_time_record, load_parameters

from helpers import config_loader, model_pipeline, update_configs_yml

# GLOABAL: Load configs parameters
params_loader = config_loader("configs.yml")
models = params_loader["models"]
hyper_tuner = models["hyper_tuner"]


# Tune model perameters
def hyper_optimizer(
    baseline: BaseEstimator,
    params_: list, 
    x: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    # Load hyperparamerer tuners
    seed_ = hyper_tuner["seed"]
    shuffle_ = hyper_tuner["shuffle"]
    num_split_ = hyper_tuner["num_split"]
    scoring_metrics = hyper_tuner["scoring_metrics"]
    path_ = hyper_tuner["path"]
    method_ = hyper_tuner["method"]
    selected_cols_ = hyper_tuner["selected_cols"]

    # Define KFlod
    kflod_cv = StratifiedKFold(
        n_splits=num_split_, random_state=seed_, shuffle=shuffle_
    )

    # Label Encoding
    label_encoding = LabelEncoder()
    y = label_encoding.fit_transform(y)

    # Parameter optimization
    pipeline = model_pipeline(baseline, params_, None)

    # Selecting optimization method
    cv_search = None
    if method_ == "randomized":
        cv_search = RandomizedSearchCV(
            pipeline,
            param_distributions=params_,
            n_iter=5,
            cv=kflod_cv,
            scoring=scoring_metrics,
            refit=scoring_metrics[1],
            random_state=seed_,
            verbose=1,
            n_jobs=-1,
        ).fit(x, y)

    elif method_ == "grid":
        cv_search = GridSearchCV(
            pipeline,
            param_grid=params_,
            cv=kflod_cv,
            scoring=scoring_metrics,
            refit=scoring_metrics[1],
            verbose=1,
            n_jobs=-1,
        ).fit(x, y)
    else:
        sys.exit("Undefined hyperprameter tuning method.")

    cv_result_df = pd.DataFrame(cv_search.cv_results_)[selected_cols_]
    cv_result_df = cv_result_df.sort_values(selected_cols_[-1])

    return cv_result_df


if __name__ == "__main__":
    print("Started hyper-param tuning ...")
    if not models["train"]["optimize"]:
        print("Not need for parameter tunning ...")
        sys.exit(0)

    print("Optimizing hyperparamters and retraining the model ...")

    # Reading params files
    path_ = hyper_tuner["path"]
    os.makedirs(path_, exist_ok=True)

    print(f"path: {path_}")

    file_input_name_ = params_loader["data"]["split"]["files"][0]
    file_input_path_ = Path(sys.argv[1]) / file_input_name_
    # print(file_input_path_)

    model_checkpoints = models["path"]
    baseline_ = models["baseline"]["path"]
    baseline_model_path_ = Path(model_checkpoints) / baseline_
    # print(baseline_model_path_)

    # Load training dataset
    dataframe = pd.read_csv(file_input_path_)
    x_all_ = dataframe[["text"]]
    y_all_ = dataframe["sentiment"]

    # Initiate hyer-tuner optimization
    model_name_ = models["train"]["model"]
    search_model_ = f"{model_name_}_model.pkl"
    # print(search_model_)

    # Check if model file exist in development stage
    try:
        model_collections = [
            str(f).split("/")[-1]
            for f in list(Path(baseline_model_path_).glob("*.pkl"))
        ]

        if not search_model_ in model_collections:
            raise ValueError(
                f"`{search_model_}` model does not exists! Please run model_dev.py script."
            )

    except ValueError as e:
        sys.exit(f"{e}")

    model_pickle_ = Path(baseline_model_path_) / search_model_
    clf = joblib.load(model_pickle_)
    params_ = [vector_params_, model_params_[model_name_]]
    

    tune_results = hyper_optimizer(clf, params_, x_all_, y_all_)
    tune_file_out_ = Path(path_) / f"optimized_params.csv"
    tune_results.to_csv(tune_file_out_, index=False)

    # Execute filtering parameter script & retraining the model
    clf_name = clf.__class__.__name__.lower()
    subprocess.call(f"./scripts/filtering_best_params.sh {path_} {clf_name}", shell=True)

    # Retraining the model using optimized parameters
    subprocess.call(f"python src/model_development/model_training.py {sys.argv[1]} src/model_checkpoints/optimized", shell=True)

    # Update config file and set optimize = False
    print("Set configs.yml optimize value to false.")
    update_configs_yml("configs.yml", "optimize", False)