from .utils import *
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# Load training dataset for model training
dataset = load_train_split("train.csv")

X_all= dataset[["text"]]
y_all = dataset[["sentiment"]]

# Tune model perameters
def hyper_optimizer(baseline: BaseEstimator, parameters: dict, cv_splits=3) -> pd.DataFrame:
    
    # pipeline Implementation
    vect = TfidfVectorizer(
        ngram_range=(1,1),
        norm="l2",
        analyzer="word",
        tokenizer=word_processor,
        preprocessor=word_processor
    )
    ct = make_column_transformer((vect, "text"), remainder="drop")
    pipeline = make_pipeline(ct, baseline)

    # Define KFlod 
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=43)
    
    # Model pipeline parameters 
    max_ngram = 4
    min_ngram = 1
    ngram_ranges = [
        (i,j) for i in range(min_ngram, max_ngram + 1) 
        for j in range(min_ngram, max_ngram + 1)]
    
    params = [{"columntransformer__tfidfvectorizer__ngram_range": ngram_ranges,
            "columntransformer__tfidfvectorizer__norm": ["l1", "l2"]}, parameters]

    # Label Encoding
    label_encoding = LabelEncoder()
    y_all_ = label_encoding.fit_transform(y_all)

    # All pipeline paramters
    model_params = {
        "cv": cv,
        "n_jobs": -1,
        "verbose": 1,
        "param_grid": params,
        "refit": "roc_auc_ovo",
        "scoring": ("f1_weighted", "roc_auc_ovo"),
    }

    # Parameter optimization
    cv_pipe = GridSearchCV(pipeline, **model_params) \
        .fit(X_all, y_all_)

    selected_cols = [
        "params", 
        "mean_test_f1_weighted", 
        "mean_test_roc_auc_ovo",
        "rank_test_roc_auc_ovo"
    ]
    
    pd.set_option("display.max_colwidth", None)
    cv_result_df = pd.DataFrame(cv_pipe.cv_results_)[selected_cols]
    cv_result_df = cv_result_df.sort_values("rank_test_roc_auc_ovo")
    
    return cv_result_df


if __name__ == "__main__":
    
    
    params = {
    "decisiontreeclassifier__criterion": ["gini", "entropy", "log_loss"],
    "decisiontreeclassifier__max_depth": [10, 50, 100, 500, 1000],
    "decisiontreeclassifier__max_features": ["sqrt", "log2"]
    }

    baseline = DecisionTreeClassifier(random_state=43)
    tune_results = hyper_optimizer(baseline, params)
    