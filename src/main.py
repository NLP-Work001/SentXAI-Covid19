import yaml
import subprocess
import json
from pathlib import Path
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
    best_params = _tune_params_loader("src/model_checkpoints/tuned")

    model_params_ = best_params["model"]
    vect_params_ = best_params["vectorizer"]

    print(model_params_)
    print(vect_params_)


    subprocess.call("python src/testing.py", shell=True)