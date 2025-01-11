import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_parameters


# Train/Test Split
def data_split(
    df: pd.DataFrame, train_out: str, test_out: str, ratio: float, seed: int
) -> tuple:
    x_all = df[["text"]]
    y_all = df[["sentiment"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=ratio, random_state=seed, stratify=y_all
    )

    # Create dataframes
    train_ = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
    test_ = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)

    # File split load
    train_.to_csv(train_out, index=False)
    test_.to_csv(test_out, index=False)

    return train_.shape[0], test_.shape[0]


# Main function
def main() -> None:
    print("Started data splits ...")
    params_loader = load_parameters("params.yml")

    parent_ = params_loader["data"]
    path_in_ = parent_["processed"]["path"]
    file_in_ = Path(path_in_) / parent_["processed"]["file"]

    path_out_ = parent_["split"]["path"]
    os.makedirs(path_out_, exist_ok=True)
    ratio_ = parent_["split"]["ratio"]
    seed_ = parent_["split"]["seed"]

    df = pd.read_csv(file_in_)
    train_out_ = Path(path_out_) / parent_["split"]["file"][0]
    test_out_ = Path(path_out_) / parent_["split"]["file"][1]

    train_size, test_size = data_split(df, train_out_, test_out_, ratio_, seed_)

    file_log = Path(path_out_) / "log.txt"
    with open(file_log, "w", encoding="utf-8") as f:
        f.write(
            f"data row-size: {df.shape[0]} \n train-size: {train_size}\n test-size: {test_size}"
        )

    print("Completed!")


if __name__ == "__main__":
    main()
