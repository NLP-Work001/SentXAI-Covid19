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
    y: pd.DataFrame,
    num_split=3,
) -> pd.DataFrame:
    # Define KFlod
    cv = StratifiedKFold(n_splits=num_split, shuffle=True, random_state=43)

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
    # CLI Arguments
    parser = ArgumentParser()
    parser.add_argument("-s", "--split", help="data/split directory.")
    parser.add_argument(
        "-d", "--date", help="retrieves datetime during script execution."
    )
    parser.add_argument("-o", "--out", help="contains model output path.")

    args = parser.parse_args()

    # Reading training data
    parameter_ = load_parameters("params.yml")
    train_in_ = parameter_["data"]["split"][0]
    file_in_ = Path(args.split) / train_in_

    nrows = 2000 # ToDo: delete after testing
    data = pd.read_csv(file_in_).head(nrows)
    x_all_ = data[["text"]]
    y_all_ = data["sentiment"]

    # Model params
    model_attrs_ = parameter_["model_attrs"]
    model_type_ = model_attrs_["tune_optimizer"]["type"]
    model_name_ = tune_params[model_type_][0]
    path_out_ = f"{args.out}/{model_name_}"
    model_param_ = tune_params[model_type_][1]

    pipe_params_ = [vector_params, model_param_]
    print("starting optimizing ..")

    # Run model optimizer
    pickle_file_ = model_attrs_["file"]
    tuned_out_ = model_attrs_["tune_optimizer"]["tuned_out"]
    date_time_ = date_time_record(args.date)
    file_name_out_ = tuned_out_ + "_" + date_time_ + ".csv"

    pickle_file_in_ = Path(path_out_) / pickle_file_
    tuning_out_ = Path(path_out_) / file_name_out_

    # print("Train Input File", train_in_)
    # print("File Input:, ", file_in_)
    # print("Model name: ", model_name_)
    # print("Path output: ", path_out_)
    print("Pickle File: ", pickle_file_)
    print("Pipeline params: ", pipe_params_)
    print("Pickle file path: ", pickle_file_in_)
    print("Tunining output path: ", tuning_out_)

    model = joblib.load(pickle_file_in_)
    num_split_ = model_attrs_["tune_optimizer"]["n_split"]
    tune_results = hyper_optimizer(model, pipe_params_, x_all_, y_all_, num_split_)
    tune_results.to_csv(tuning_out_, index=False)
