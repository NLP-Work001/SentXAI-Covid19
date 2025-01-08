from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
import joblib
from utils import model_pipeline, calculate_metric_score, load_train_split
import json

def __optimized_paramters(path: str) -> dict:
        params_ = {}
        json_path = Path(path) /"best_params.json"

        with open(json_path, "r") as f:
            data = json.load(f)

        # Access optimized model parameters
        for c in data:
            for i, j in c.items():
                params_[i] = {key: tuple(value) if isinstance(value, list) else value for key, value in j["params"].items()}
        return params_


def __training(model: BaseEstimator, vector_params: dict) -> None:
    # Load training & testing dataset
    _train = load_train_split("train.csv")
    _test = load_train_split("test.csv")

    x_train_ = _train[["text"]]
    y_train_ = _train[["sentiment"]]

    x_test_ = _test[["text"]]
    y_test_ = _test["sentiment"]

    # Label Encoding
    binarizer = LabelBinarizer(sparse_output=False)

    # label_encoding = LabelEncoder()
    y_train_ = binarizer.fit_transform(y_train_)
    y_test_ = binarizer.transform(y_test_)

    # training model
    best_model = DecisionTreeClassifier(
        criterion="entropy", max_depth=10, max_features="sqrt", random_state=43
    )

    norm = vector_params["norm"]
    ngram_range = vector_params["ngram_range"]

    model = model_pipeline(best_model, ngram_range, norm).fit(
        x_train_, y_train_.argmax(axis=1)
    )

    train_metric_scores = calculate_metric_score(
        model, x_train_, y_train_.argmax(axis=1)
    )
    test_metric_scores = calculate_metric_score(model, x_test_, y_test_.argmax(axis=1))

    print("Training Scores: ", train_metric_scores)
    print("Testing Scores: ", test_metric_scores)

def main() -> None:
    path = "models/logisticregression"
    optimized_params = __optimized_paramters(path)

    # Re-training model
    vectorizer_params = optimized_params["vectorizer"]

    pickle_file_ = Path(path) / "model.pkl"
    model = joblib.load(pickle_file_)
    model.set_params(**optimized_params["model"])

    __training(model, vectorizer_params)

# ToDo: Log the parameters into `evaluation folder with models directory`
# ToDo: Log the final pickle file for predictions

if __name__ == "__main__":
    main()
