import os
from argparse import ArgumentParser
from pathlib import Path

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
    y: pd.Series,
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
    params_loader = load_parameters("config.yml")
    # fine_tune:
    #     model: logistic
    #     n_splits: 3
    #     path: tuned/

    # Reading data file
    # parent_ = params_loader["data"]
    # path_in_ = parent_["split"]["path"]
    # train_in_ = Path(path_in_) / parent_["split"]["file"][0]
    parent_split_ = params_loader["data"]
    path_split_ = parent_split_["split"]
    path_in_ = path_split_["path"]
    train_in_ = Path(path_in_) / path_split_["files"][0]

    # Fine-Tuned utils
    models_stage = params_loader["models"]
    parent_tune_ = models_stage["fine_tune"]
    model_name_ = parent_tune_["model"]
    path_out_ = parent_tune_["path"]
    # tune_model_out_ = Path(models_stage["dev"]["path"]).parent / path_out_ / model_name_
    # print(tune_model_out_)
    # print(model_name_)
    # print(num_split_)
    # print(tune_model_out_)

    # command-Line arguments
    parser = ArgumentParser()
    parser.add_argument("-d", "--date", help="Recorded date during runtime execution.")
    parser.add_argument("-o", "--out", help="Output model directory")
    args = parser.parse_args()

    date_time = date_time_record(args.date)
    tune_model_out_ = args.out
    os.makedirs(tune_model_out_, exist_ok=True)

    # Load training dataset
    dataframe = pd.read_csv(train_in_)
    x_all_ = dataframe[["text"]]
    y_all_ = dataframe["sentiment"]

    # Initiate hyer-tuner optimization
    seed_ = models_stage["seed"]
    num_split_ = parent_tune_["n_splits"]
    search_model_ = f"{model_name_}_model.pkl"

    # Check if model file exist in development stage
    try:
        models = [
            str(f).split("/")[-1]
            for f in list(Path(models_stage["dev"]["path"]).glob("*.pkl"))
        ]
        # print(models)
        if not search_model_ in models:
            raise ValueError(
                f"`{search_model_}` model does not exists! Please run model_dev.py script."
            )

    except ValueError as e:
        sys.exit(f"{e}")

    model_pickle_ = Path(models_stage["dev"]["path"]) / search_model_
    clf = joblib.load(model_pickle_)

    model_params_ = tune_params[model_name_]
    pipe_params_ = [vector_params, model_params_]

    tune_results = hyper_optimizer(clf, pipe_params_, x_all_, y_all_, num_split_, seed_)

    file_out_ = f"optimized_params_{date_time}.csv"
    tune_file_out_ = Path(tune_model_out_) / file_out_
    # print(file_out_)
    # print(tune_file_out_)

    tune_results.to_csv(tune_file_out_, index=False)

    model_out = clf.__class__.__name__.lower()
    os.environ["FOLDER_NAME"] = model_name_
    os.environ["MODEL_NAME"] = clf.__class__.__name__.lower()

    print(f"folder: {os.environ.get('FOLDER_NAME')}")
    print(f"moodel: {os.environ.get('MODEL_NAME')}")

    # print(seed_)
    # print(num_split_)
    # print(train_in_)
    # print(model.__class__.__name__)
    # print(tune_path_out_)
    # print(tune_file_out_)
    # print(pickle_in_)
    # print(pipe_params_)
