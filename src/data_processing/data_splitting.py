import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(str("src/helpers"))
from utils import load_parameters
from helpers import config_loader

# Train/Test Split
def data_split(
    df: pd.DataFrame, train_out: str, test_out: str, ratio: float, seed: int
) -> tuple:
    x_all = df[["text"]]
    y_all = df["sentiment"]
    
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
    params_loader = config_loader("configs.yml")

    parent_ = params_loader["data"]

    path_out_ = sys.argv[2]
    os.makedirs(path_out_, exist_ok=True)
    path_split_ = parent_["split"]
    ratio_ = path_split_["ratio"]
    seed_ = path_split_["seed"]

    # Read processed data file
    file_path_ = Path(sys.argv[1]) / parent_["processed"]["file"]
    df = pd.read_csv(file_path_).dropna()
 
    # Setup output files
    train_out_ = Path(path_out_) / path_split_["files"][0]
    test_out_ = Path(path_out_) / path_split_["files"][1]

    # Train/test split
    train_size, test_size = data_split(df, train_out_, test_out_, ratio_, seed_)

    # Log file sizes
    file_log = Path(path_out_) / "log.txt"
    with open(file_log, "w", encoding="utf-8") as f:
        f.write(
            f"Data row-size: {df.shape[0]}\ntrain-size: {train_size}\ntest-size: {test_size}"
        )

    print("Completed!")
    

if __name__ == "__main__":
    main()
