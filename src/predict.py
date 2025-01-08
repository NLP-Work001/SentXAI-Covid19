from pathlib import Path

import joblib
import pandas as pd

from utils import get_output_label, load_parameters

if __name__ == "__main__":
    param_loader = load_parameters("params.yml")
    training_arg = param_loader["training"]
    model_name = training_arg["model"]

    eval_out_ = Path(training_arg["path_out"]) / model_name

    text: str = "spain controlling enter grocery store"
    input_df = pd.DataFrame([[text]], columns=["text"])
    pickle_file_ = Path(eval_out_) / "model.pkl"
    model = joblib.load(pickle_file_)

    encoder_file_ = Path(eval_out_) / "encoder_binarizer.pkl"
    encoding = joblib.load(encoder_file_)

    predicted = model.predict_proba(input_df).argmax(axis=1)

    label = get_output_label(encoding, predicted[0])
    print("Input: ", text)
    print("Predicted: ", label)
