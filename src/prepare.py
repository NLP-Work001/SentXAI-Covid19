import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load processed dataset
filename = "covid19_tweets.csv"
folder_dir = "data/processed"
file_path = Path(f"{folder_dir}") / filename


df = pd.read_csv(file_path) \
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


def main() -> None:
    
    X_train, X_test, y_train, y_test = split_data(df)

    # Create dataframes
    train_ = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_ = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    
    # Output file names
    split_output_dir = "data/splits"
    train_filename = Path(f"{split_output_dir}/train.csv")
    test_filename = Path(f"{split_output_dir}/test.csv")

    # File split load
    train_.to_csv(train_filename, index=False)
    test_.to_csv(test_filename, index=False)

if __name__ == "__main__":
    main()