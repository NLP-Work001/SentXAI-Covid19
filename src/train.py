from .utils import *
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer

# ML algorithms
from sklearn.tree import DecisionTreeClassifier

# Model training pipeline
def _model_pipeline(baseline: BaseEstimator, vect_ngram_=(1,1), vect_norm_="l2"):
    
    vect = TfidfVectorizer(
        ngram_range=vect_ngram_,
        norm=vect_norm_,
        analyzer="word",
        tokenizer=word_processor,
        preprocessor=word_processor
    )
    ct = make_column_transformer((vect, "text"), remainder="drop")
    pipeline = make_pipeline(ct, baseline)
    
    return pipeline


def main() -> None:
    # Load training & testing dataset
    _train = load_train_split("train.csv")
    _test = load_train_split("test.csv")

    X_train_= _train[["text"]]
    y_train_ = _train[["sentiment"]]

    X_test_= _test[["text"]]
    y_test_ = _test[["sentiment"]]

    # Label Encoding
    binarizer = LabelBinarizer(sparse_output=False)
    
    # label_encoding = LabelEncoder()
    y_train_ = binarizer.fit_transform(y_train_)
    y_test_ = binarizer.transform(y_test_)

    # training model
    best_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features='sqrt', random_state=43)
    model = _model_pipeline(best_model, (4,4), "l1").fit(X_train_, y_train_.argmax(axis=1))

    train_metric_scores = calculate_model_metrics(model, X_train_, y_train_.argmax(axis=1))
    test_metric_scores = calculate_model_metrics(model, X_test_, y_test_.argmax(axis=1))

    print("Training Scores: ", train_metric_scores)
    print("Testing Scores: ", test_metric_scores)
    
if __name__ == "__main__":
    main()