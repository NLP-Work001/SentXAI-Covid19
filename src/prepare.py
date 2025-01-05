import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_parameters

# Train/Test Split
def split_data(data: pd.DataFrame, test_size=0.2, random_state=43):
    x_all = data[["text"]]
    y_all = data[["sentiment"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
    )

    return x_train, x_test, y_train, y_test


# Data loading
def loading_data(file_path: str, train_out: str, test_out: str) -> None:
    df = pd.read_csv(file_path).dropna().reset_index(drop=True).drop("tweet", axis=1)
    x_train, x_test, y_train, y_test = split_data(df)

    # Create dataframes
    train_ = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
    test_ = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)

    # File split load
    train_.to_csv(train_out, index=False)
    test_.to_csv(test_out, index=False)
    print("End data split ..")


# Main function
def main() -> None:
    # Load parameters
    parameters = load_parameters("params.yml")

    # File Inputs
    FILE_NAME = parameters["data"]["processed"][0]
    INPUT_PATH = Path(f"{sys.argv[1]}") / FILE_NAME

    # File outputs
    FILE_TRAIN = parameters["data"]["split"][0]
    FILE_TEST = parameters["data"]["split"][1]

    TRAIN_OUT = Path(f"{sys.argv[2]}") / FILE_TRAIN
    TEST_OUT = Path(f"{sys.argv[2]}") / FILE_TEST

    print("File input path: ", INPUT_PATH)
    print("File train output path: ", TRAIN_OUT)
    print("File test output path: ", TEST_OUT)

    print("Started data split ...")
    loading_data(INPUT_PATH, TRAIN_OUT, TEST_OUT)


if __name__ == "__main__":
    main()
