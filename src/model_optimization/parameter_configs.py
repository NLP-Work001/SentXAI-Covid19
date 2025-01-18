import numpy as np

# Model based parameters
model_params_ = {
    "logistic": {
        "logisticregression__C": [round(c, 2) for c in np.linspace(0.01, 2, 4)],
        "logisticregression__solver": ["newton-cg", "sag", "saga", "lbfgs"],
        "logisticregression__penalty": ["l2"],
        "logisticregression__max_iter": [c for c in range(1000, 2000, 250)],
    },
    "tree": {
        "decisiontreeclassifier__criterion": ["gini", "entropy", "log_loss"],
        "decisiontreeclassifier__max_depth": [10, 50, 100, 500, 1000],
        "decisiontreeclassifier__max_features": ["sqrt", "log2"],
    },
}

# Vectorizer based params
MAX_NGRAM_ = 4
__ngram_ranges = [
    (i, j) for i in range(1, MAX_NGRAM_ + 1) for j in range(1, MAX_NGRAM_ + 1) if i <= j
]
vector_params_ = {
    "columntransformer__tfidfvectorizer__ngram_range": __ngram_ranges,
    "columntransformer__tfidfvectorizer__norm": ["l1", "l2"],
}
