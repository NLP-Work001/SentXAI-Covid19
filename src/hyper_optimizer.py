import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from utils import model_pipeline, load_train_split

# Load training dataset for model training
dataset = load_train_split("train.csv")

X_all = dataset[["text"]]
y_all = dataset[["sentiment"]]


# Tune model perameters
def hyper_optimizer(
    baseline: BaseEstimator, pipeline_params: list, cv_splits=3
) -> pd.DataFrame:
    # Define KFlod
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=43)

    # Label Encoding
    label_encoding = LabelEncoder()
    y_all_ = label_encoding.fit_transform(y_all)

    # All pipeline paramters
    search_params = {
        "cv": cv,
        "n_jobs": -1,
        "verbose": 1,
        "param_grid": pipeline_params,
        "refit": "roc_auc_ovo",
        "scoring": ("f1_weighted", "roc_auc_ovo"),
    }

    # Parameter optimization
    pipeline = model_pipeline(baseline, pipeline_params, None)
    cv_pipe = GridSearchCV(pipeline, **search_params).fit(X_all, y_all_)

    selected_cols = [
        "params",
        "mean_test_f1_weighted",
        "mean_test_roc_auc_ovo",
        "rank_test_roc_auc_ovo",
    ]

    pd.set_option("display.max_colwidth", None)
    cv_result_df = pd.DataFrame(cv_pipe.cv_results_)[selected_cols]
    cv_result_df = cv_result_df.sort_values("rank_test_roc_auc_ovo")

    return cv_result_df


if __name__ == "__main__":
    # Vectorizer params
    MAX_NGRAM_ = 4
    ngram_ranges = [
        (i, j)
        for i in range(1, MAX_NGRAM_ + 1)
        for j in range(1, MAX_NGRAM_ + 1)
        if i <= j
    ]
    vector_specific = {
        "columntransformer__tfidfvectorizer__ngram_range": ngram_ranges,
        "columntransformer__tfidfvectorizer__norm": ["l1", "l2"],
    }

    # Model params
    model_specific = {
        "decisiontreeclassifier__criterion": ["gini", "entropy", "log_loss"],
        "decisiontreeclassifier__max_depth": [10, 50, 100, 500, 1000],
        "decisiontreeclassifier__max_features": ["sqrt", "log2"],
    }

    tune_params = [vector_specific, model_specific]
    print("starting optimizing ..")

    # Run model optimizer
    model = DecisionTreeClassifier(random_state=43)
    tune_results = hyper_optimizer(model, tune_params)
    print(tune_results)
