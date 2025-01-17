from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
import json
from utils import get_output_label, load_parameters
from process import text_preprocessing

# @lru_cache(maxsize=None)
def predict_fn(text: str):
    param_loader = load_parameters("params.yml")

    parent_path_ = param_loader["dev"]["path"]
    child_path_ = param_loader["dev"]["train"]["path"]
    type_  = param_loader["dev"]["train"]["model"]
    pickle_ = param_loader["dev"]["file"]
    pickle_loader_ = Path(f"{parent_path_}/{child_path_}/{type_}") / pickle_

    encoder_ = param_loader["dev"]["train"]["encoder"]
    pickle_encoder = Path(f"{parent_path_}/{child_path_}/{type_}") / encoder_

    # Model Predictions
    input_df = pd.DataFrame([[text]], columns=["tweet"])
    input = text_preprocessing(input_df)
    print(input)

    model = joblib.load(pickle_loader_)
    encoding = joblib.load(pickle_encoder)

    predicted = model.predict_proba(input[["text"]]).argmax(axis=1)

    return {"predicted": get_output_label(encoding, predicted[0]), "processed": input["text"].values[0]}
    # return input["text"].values


if __name__ == "__main__":

    # print("Started prediction ...")
    # text = "or buy gift certificates for others who may be in self-quarantine. a little online shopping will pass the time and keep local businesses alive. plus, more books! #coronavirus #books"
    #
    # print("Input: ", text)
    # print("Predicted: ", predict_fn(text))
    path = "models/trained/logistic/metrics.json"
    with open(path, "rb") as f:
        data = json.load(f)
        print(pd.DataFrame(data))
