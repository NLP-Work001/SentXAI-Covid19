"""This script performs model training and calcualte the score metrics.
It uses a scikit-learn pipeline that includes both the vectorizer and
the selected mdoel.
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier

from utils import model_pipeline, calculate_metric_score, load_train_split


def main() -> None:
    """This function encapsulates the script other functions"""
    # Load training & testing dataset
    _train = load_train_split("train.csv")
    _test = load_train_split("test.csv")

    x_train_ = _train[["text"]]
    y_train_ = _train[["sentiment"]]

    x_test_ = _test[["text"]]
    y_test_ = _test[["sentiment"]]

    # Label Encoding
    binarizer = LabelBinarizer(sparse_output=False)

    # label_encoding = LabelEncoder()
    y_train_ = binarizer.fit_transform(y_train_)
    y_test_ = binarizer.transform(y_test_)

    # training model
    best_model = DecisionTreeClassifier(
        criterion="entropy", max_depth=10, max_features="sqrt", random_state=43
    )
    model = model_pipeline(best_model, (4, 4), "l1").fit(
        x_train_, y_train_.argmax(axis=1)
    )

    train_metric_scores = calculate_metric_score(
        model, x_train_, y_train_.argmax(axis=1)
    )
    test_metric_scores = calculate_metric_score(model, x_test_, y_test_.argmax(axis=1))

    print("Training Scores: ", train_metric_scores)
    print("Testing Scores: ", test_metric_scores)


if __name__ == "__main__":
    main()
