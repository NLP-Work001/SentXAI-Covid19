import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load processed dataset
FILE_DIR = "data/processed"
FILE_PATH = Path(f"{FILE_DIR}/covid19_tweets.csv")

pd.set_option("display.max_colwidth", None)

df = pd.read_csv(FILE_PATH) \
    .dropna() \
    .reset_index(drop=True) \
    .drop("tweet", axis=1)

# Train/Test Split
def split_data(df, test_size=0.2, random_state=43):
    X_all = df[["text"]]
    y_all = df[["sentiment"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
    )

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(df)

# Create dataframes
train_ = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
test_ = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

if __name__ == "__main__":
    # Output file names
    OUTPUT_DIR = "data/splits"
    TRAIN_FILE = Path(f"{OUTPUT_DIR}/train.csv")
    TEST_FILE = Path(f"{OUTPUT_DIR}/test.csv")

    # File split load
    train_.to_csv(TRAIN_FILE, index=False)
    test_.to_csv(TEST_FILE, index=False)