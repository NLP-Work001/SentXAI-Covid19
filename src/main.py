import yaml
import subprocess
import json
from pathlib import Path
import joblib
from itertools import product
import sys
sys.path.append(str("src/helpers"))
from utils import (
    calculate_model_metrics,
    date_time_record,
    get_output_label,
    model_pipeline,
)
import re
import os
from mlflow import MlflowClient

from sklearn.base import BaseEstimator

def config_loader(file_name: str) -> dict:
    with open(file_name, "r") as f:
        params = yaml.safe_load(f)
    return params

def update_configs_yml(file_path: str, key: str, new_value: bool) -> None:
    
    configs = config_loader(file_path)
    get_retrain = configs["models"]["retrain"]
    
    if key in get_retrain:
        get_retrain["is_retrain"] = new_value
    else:
        print(f"Key `{key}` not found in the YML file.")
        return
    
    with open("configs.yml", "w") as f:
        yaml.dump(configs, f, default_flow_style=False)
        
        print(f"Updated `{key}` to `{new_value}` in {file_path}.")
 

def _tune_params_loader(path: str) -> dict:
    params_ = {}
    file_in_ = Path(path) / "best_params.json"

    with open(file_in_, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, j in data.items():
        params_[i] = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in j["params"].items()
        }
    params_["model"] = data["model"]["params"]

    return params_

if __name__ == "__main__":
    # # update_configs_yml("configs.yml", "is_retrain", False)
    # model_results = {
    #         "train": {
    #             "f1": 0.89,
    #             "roc_auc": 0.99
    #         },
    #         "test": {
    #             "f1": 0.56,
    #             "roc_auc": 0.75
    #         }
    #     }

    # columns1 = next(iter([list(c.keys()) for c in model_results.values()]))
    # print(columns1)
    # name = "Lana"
    # subprocess.call("chmod +x scripts/test.sh", shell=True)
    # subprocess.call(f"./scripts/test.sh {name}", shell=True)
    # best_params = _tune_params_loader("src/model_checkpoints/tuned")

    # model_params_ = best_params["model"]
    # vect_params_ = best_params["vectorizer"]

    # print(model_params_)
    # print(vect_params_)


    # subprocess.call("python src/testing.py", shell=True)
    # path = Path("src/model_checkpoints/trained")
    # artifact_names = [Path(path/c) for c in os.listdir(path) if ".json" not in c and "model" not in c]
    # # for c in artifact_names:
    # #     mlflow.log_artifact(file_path, artifact_path="logistic")

    # print(artifact_names)
    # metrics = {
    #         "train": {
    #             "f1": 0.897,
    #             "roc_auc": 0.987
    #         },
    #         "test": {
    #             "f1": 0.59,
    #             "roc_auc": 0.768
    #         }
    #     }
    # def formated_metrics(metrics: dict) -> dict: 
    #     metrics = {f"{k}-{i}": j for k, v in metrics.items() for i, j in v.items()}
    #     return  metrics

    # print(formated_metrics(metrics))
    # model = joblib.load("src/model_checkpoints/trained/sk_pipeline.pkl")
    # model_name = list(model.named_steps.keys())[-1]
    # print(model_name)
    # Images logged as figutres
    # csv and pickle files are logged as artifacts
    # Model file is logged directly to mflow env
    # Cross validation metrics are appended into trained logged model
    # Hyperparamerers and metrics are logged separately from trained model with metrics and pickle files
    # Optimized model is trained using hyperparamters that were optimized.
    # class_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    # comb = list(product(class_weights, repeat=3))
    # # print(comb)

    # print([c for c in comb if sum(c) == 1])

    # def default_pipeline_params(pipeline: BaseEstimator, model_name: str) -> dict:
    #     default_model_params = pipeline.named_steps[model_name].get_params()
    #     default_vect_params = pipeline.named_steps["columntransformer"].transformers_[0][1].get_params()

    #     pipeline_params = {"tfidf": default_vect_params, "model": default_model_params}
    #     return pipeline_params
    # import pandas as pd
    # # print(default_pipeline_params(model, "logisticregression"))
    # with  open("data/split/data_info.json", "r", encoding="utf-8") as f:
    #     data_info = json.load(f)

    # # print(data_info)
    # data_sample = pd.read_csv("data/split/train.csv").sample(n=5).reset_index(drop=True)

    # print(data_sample[["text"]])
    client = MlflowClient()

    # print(help(client.search_registered_models))
   # Get all registered models and order them by ascending order of the names
    results = client.search_registered_models(order_by=["metric.test-roc_auc ASC"])
    print("-" * 80)
    for res in results:
        for mv in res.latest_versions:
            print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

