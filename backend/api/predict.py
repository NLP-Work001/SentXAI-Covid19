import sys
import joblib
import pandas as pd
from pathlib import Path
from django.conf import settings
sys.path.append(str(settings.SCRIPT_PATH))
sys.path.append(str(settings.MODEL_PATH))
from utils import get_output_label
from process import text_preprocessing


def predict_fn(text: str):
    
    pickle_loader_ = "model.pkl"
    pickle_encoder = "label_encoder.pkl"

    input_df = pd.DataFrame([[text]], columns=["tweet"])
    input = text_preprocessing(input_df)

    model = joblib.load(pickle_loader_)
    encoding = joblib.load(pickle_encoder)

    predicted = model.predict_proba(input[["text"]]).argmax(axis=1)

    return {"processed": input["text"].values[0], "predicted": get_output_label(encoding, predicted[0]),}
