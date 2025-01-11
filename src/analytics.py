import os
import warnings
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import load_parameters

warnings.filterwarnings("ignore")
# plt.style.use("ggplot")


# Load processed data file
def reading_file(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)

    data.columns = [c.lower() for c in data.columns]
    data["tokens"] = data["text"].str.split()
    data = data[["text", "tokens", "sentiment"]]

    return data


# Explode tokens and create a DataFrame
def token_word_count(df: pd.DataFrame) -> pd.DataFrame:
    tokens = df.tokens.explode().reset_index(drop=True).values
    token_df = pd.DataFrame(tokens, columns=["word"])

    # Group and count word occurrences
    word_count = (
        token_df.groupby("word", as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )

    return word_count


# Words Frequency function
def plot_word_frequency(df: pd.DataFrame, file_out: str, topn=25) -> None:
    count_df = token_word_count(df)

    # Create bar chart
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(count_df[:topn], x="word", y="size", color="red", gap=0.2, ax=ax)

    # Customize plot
    ax.set_title("Words Occurrence", fontsize=12, alpha=0.8)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Word", fontsize=12)
    plt.xticks(rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    data_size = df.shape[0]
    file_name = Path(file_out) / f"word_frequency_barplot_{data_size}.png"

    fig.savefig(file_name, dpi=100)


# Plot sentence lengths (tokens per sentence)
def plot_sentence_token_length(df: pd.DataFrame, file_out: str) -> None:
    sentence_lengths = df["tokens"].str.len().dropna()

    # Create a histogram of sentence lengths
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 3.5))

    sns.histplot(x=sentence_lengths, bins=12, color="blue", alpha=0.7, ax=ax)

    # Set axis labels and title
    ax.set_title("Tweet Sentence Distribution", fontsize=12, alpha=0.8)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xlabel("Number of Tokens (Sentence Length)", fontsize=12)

    # Add grid and adjust layout
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    data_size = df.shape[0]
    file_name = Path(file_out) / f"sentence_len_histogram_{data_size}.png"

    fig.savefig(file_name, dpi=100)


# Plot sentiment distributions
def plot_label_distr(df: pd.DataFrame, file_out: str, palette="colorblind") -> None:
    sent_df = df["sentiment"].value_counts().reset_index(name="size")
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.barplot(
        sent_df,
        x="sentiment",
        y="size",
        hue="sentiment",
        palette=palette,
        gap=0.5,
        ax=ax,
    )

    # Axis Labels and Title
    ax.set_title("Sentiment Distribution", fontsize=12, alpha=0.8)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("")
    plt.xticks(rotation=45)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    data_size = df.shape[0]
    file_name = Path(file_out) / f"sentiment_label_barplot_{data_size}.png"

    fig.savefig(file_name, dpi=100)


def main() -> None:
    print("Started analysis ...")
    params_loader = load_parameters("params.yml")

    parent_ = params_loader["data"]
    path_in_ = parent_["processed"]["path"]
    file_in_ = Path(path_in_) / parent_["processed"]["file"]

    path_out_ = parent_["analysis"]["path"]
    os.makedirs(path_out_, exist_ok=True)

    # Reading file
    dataframe = reading_file(file_in_)
    # file_out_ = Path(path_out_) / f"plot_name_{dataframe.shape[0]}.png"

    # print(file_in_)
    # print(file_out_)

    # Log plots
    plot_label_distr(dataframe, path_out_)
    plot_sentence_token_length(dataframe, path_out_)
    plot_word_frequency(dataframe, path_out_)

    print("Completed!")


if __name__ == "__main__":
    main()
