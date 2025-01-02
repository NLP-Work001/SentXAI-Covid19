from .utils import *
from sklearn.tree import DecisionTreeClassifier


# Barplots for metric comparisons
def plot_cv_scores(scores: dict) -> None:
    
    filter_metric = [
        (c.split('_')[0], "_".join(c.split("_")[1:]), k)
        if len(c.split('_')) > 2 else (*c.split('_'), k) 
        for c, k in scores.items() if "time" not in c
    ]

    cols = ["cv", "x", "y"]
    metric_df = pd.DataFrame(filter_metric, columns=cols) \
            .sort_values("x", ascending=False)

    # Seaborn plot
    _, ax = plt.subplots(figsize=(10, 4))
    plt.style.use("ggplot")
    
    sns.barplot(metric_df, x="x", y="y", width=0.5, hue="cv", palette="colorblind", ax=ax)
    # Axis Labels and Title
    ax.set_title("cross validation metrics", fontsize=12, alpha=0.8)
    ax.set_ylabel("score", fontsize=12)
    ax.set_xlabel("")
    plt.legend(title="")

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

def cross_validation_func(baseline: BaseEstimator, X: pd.DataFrame, y: pd.DataFrame) -> dict:
    
    label_encoding = LabelEncoder()
    y_all_ = label_encoding.fit_transform(y)

    # pipeline Implementation
    vect = TfidfVectorizer(
        analyzer="word",
        tokenizer=word_processor,
        preprocessor=word_processor
    )

    ct = make_column_transformer((vect, "text"), remainder="drop")
    pipeline = make_pipeline(ct, baseline)

    # Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

    train_scores = []
    val_scores = []
    time_start = time.time()
    time_lapsed = None

    for idx, (train_idx, val_idx) in tqdm(enumerate(cv.split(X, y_all))):

        print(" Iteration Count:", idx+1)
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y_all_[train_idx], y_all_[val_idx]

        pipeline.fit(X_train_fold, y_train_fold)

        train_score = calculate_model_metrics(pipeline, X_train_fold, y_train_fold)
        val_score = calculate_model_metrics(pipeline, X_val_fold, y_val_fold)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        time_lapsed = time.time() - time_start

    # Training time-lapsed
    seconds = np.round(time_lapsed)
    total_time = str(timedelta(seconds=seconds))

    # Cross-validated metric scores
    training_score = calculate_average_cv(train_scores)
    validation_score = calculate_average_cv(val_scores, "val")
    metric_scores = dict(**training_score, **validation_score, time_lapsed=total_time)
    
    return metric_scores

if __name__ == "__main__":
    
    # Load training dataset for model training
    dataset = load_train_split("train.csv")

    X_all= dataset[["text"]]
    y_all = dataset[["sentiment"]]
    
    # Training 
    baseline = DecisionTreeClassifier(random_state=43)
    knn_scores = cross_validation_func(baseline, X_all, y_all)
    
    # Save plot
    plot_cv_scores(knn_scores)