import json
import os
from argparse import ArgumentParser
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier

from utils import (
    calculate_metric_score,
    load_parameters,
    load_train_split,
    model_pipeline,
)


def __optimized_params_loader(path: str) -> dict:
    params_ = {}
    json_path = Path(path) / "best_params.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Access optimized model parameters
    for c in data:
        for i, j in c.items():
            params_[i] = {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in j["params"].items()
            }
    return params_


def __training(baseline_model: BaseEstimator, vector_params: dict, out_: str) -> None:
    # ToDo: Remove nrows after testing
    # Load training & testing dataset
    _train = load_train_split("train.csv").head(3000)
    _test = load_train_split("test.csv").head(1000)
    print("train shape: ", _train.shape)
    print("test shape: ", _test.shape)

    x_train_ = _train[["text"]]
    y_train_ = _train["sentiment"]

    x_test_ = _test[["text"]]
    y_test_ = _test["sentiment"]

    # Label Encoding
    binarizer = LabelBinarizer(sparse_output=False)

    # label_encoding = LabelEncoder()
    y_train_ = binarizer.fit_transform(y_train_)
    y_test_ = binarizer.transform(y_test_)

    binarizer_out_ = Path(out_) / "encoder_binarizer.pkl"
    joblib.dump(binarizer, binarizer_out_)
    # training model

    norm = vector_params["norm"]
    ngram_range = vector_params["ngram_range"]

    model = model_pipeline(baseline_model, ngram_range, norm).fit(
        x_train_, y_train_.argmax(axis=1)
    )

    train_scores = calculate_metric_score(model, x_train_, y_train_.argmax(axis=1))
    test_scores = calculate_metric_score(model, x_test_, y_test_.argmax(axis=1))

    scores = {"train": train_scores, "test": test_scores}
    score_out_ = Path(out_) / "metric_scores.json"

    print("Writing scores to file name: ", score_out_)
    with open(score_out_, "w", encoding='utf-8') as f:
        json.dump(scores, f)

    model_out_ = Path(out_) / "model.pkl"
    joblib.dump(model, model_out_)
    print("Write model pipeline pickle file as: ", model_out_)

    print(f"Trained model saved to: `{out_}`")
    print("Training Scores: ", train_scores)
    print("Testing Scores: ", test_scores)


def main() -> None:
    param_loader = load_parameters("params.yml")
    training_arg = param_loader["training"]
    model_name = training_arg["model"]

    path_in_ = Path(training_arg["path_in"]) / model_name
    eval_out_ = Path(training_arg["path_out"]) / model_name
    os.makedirs(eval_out_, exist_ok=True)

    optimized_params = __optimized_params_loader(path_in_)

    # Re-training model
    vectorizer_params = optimized_params["vectorizer"]

    pickle_file_ = Path(path_in_) / "model.pkl"
    model = joblib.load(pickle_file_)
    model.set_params(**optimized_params["model"])
    
    __training(model, vectorizer_params, eval_out_)


# ToDo: Log the parameters into `evaluation folder with models directory`
# ToDo: Log the final pickle file for predictions


if __name__ == "__main__":
    main()
