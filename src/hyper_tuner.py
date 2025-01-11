from argparse import ArgumentParser
from pathlib import Path
import os
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from tuner_utils import tune_params, vector_params
from utils import date_time_record, load_parameters, model_pipeline


# Tune model perameters
def hyper_optimizer(
    baseline: BaseEstimator,
    pipeline_params: list,
    x: pd.DataFrame,
    y: pd.DataFrame,
    num_split: float,
    seed: int,
) -> pd.DataFrame:
    # Define KFlod
    cv = StratifiedKFold(n_splits=num_split, random_state=seed, shuffle=True)

    # Label Encoding
    label_encoding = LabelEncoder()
    y = label_encoding.fit_transform(y)

    # Parameter optimization
    pipeline = model_pipeline(baseline, pipeline_params, None)
    cv_pipe = GridSearchCV(
        pipeline,
        param_grid=pipeline_params,
        cv=cv,
        scoring=("f1_weighted", "roc_auc_ovo"),
        refit="roc_auc_ovo",
        verbose=1,
        n_jobs=-1,
    ).fit(x, y)

    selected_cols = [
        "params",
        "mean_test_f1_weighted",
        "mean_test_roc_auc_ovo",
        "rank_test_roc_auc_ovo",
    ]

    cv_result_df = pd.DataFrame(cv_pipe.cv_results_)[selected_cols]
    cv_result_df = cv_result_df.sort_values("rank_test_roc_auc_ovo")

    return cv_result_df


if __name__ == "__main__":
    print("Started hyper-param tuning ...")
    params_loader = load_parameters("params.yml")

    # Reading data file
    parent_ = params_loader["data"]
    path_in_ = parent_["split"]["path"]
    train_in_ = Path(path_in_) / parent_["split"]["file"][0]

    # Fine-Tuned utils
    dev = params_loader["dev"]
    dev_path = dev["path"]
    cross_val_path = dev["cross-valid"]["path"]  # arg1 in a function
    tune = dev["fine-tune"]  # arg2 in a function
    tune_path_ = tune["path"]
    tune_model = tune["model"]

    cv_path_in_ = Path(f"{dev_path}/{cross_val_path}/{tune_model}")
    tune_path_out_ = Path(f"{dev_path}/{tune_path_}/{tune_model}")
    os.makedirs(tune_path_out_, exist_ok=True)
    
    # command-Line arguments
    parser = ArgumentParser()
    parser.add_argument("-d", "--date", help="Recorded date during runtime execution.")

    args = parser.parse_args()

    date_time = date_time_record(args.date)
    file_out_ = f"optimized_params_{date_time}.csv"
    tune_file_out_ = Path(tune_path_out_) / file_out_
    
    # Load training dataset
    dataframe = pd.read_csv(train_in_)
    x_all_ = dataframe[["text"]]
    y_all_ = dataframe["sentiment"]

    # Initiate hyer-tuner optimization
    num_split_ = tune["n_split"]
    seed_ = parent_["split"]["seed"]
    pickle_in_ = Path(cv_path_in_) / dev["file"]
    model = joblib.load(pickle_in_)
    model_param_ = tune_params[tune_model]
    pipe_params_ = [vector_params, model_param_]

    tune_results = hyper_optimizer(
        model, pipe_params_, x_all_, y_all_, num_split_, seed_
    )
    tune_results.to_csv(tune_file_out_, index=False)

    # print(seed_)
    # print(num_split_)
    # print(train_in_)
    # print(model.__class__.__name__)
    # print(tune_path_out_)
    # print(tune_file_out_)
    # print(pickle_in_)
    # print(pipe_params_)
