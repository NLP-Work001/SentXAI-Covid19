import re
import sys
import string
import warnings
from pathlib import Path
import os
import contractions
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
sys.path.append(str("src/helpers"))

from utils import load_parameters
warnings.filterwarnings("ignore")

# Loading data method
def __reading_file(folder: str) -> pd.DataFrame:
    df = pd.DataFrame()
    for path in list(folder.glob("*.csv")):
        data = pd.read_csv(path, encoding="latin1")
        df = (
            pd.concat([df, data], axis=0)
            .sample(frac=1, random_state=43)
            .reset_index(drop=True)
        )

    return df


# Consolidate negative and postive sentiments
def __consolidate_sentiment(sentiment: str) -> str:
    if sentiment == "extremely positive":
        return "positive"

    elif sentiment == "extremely negative":
        return "negative"
    else:
        return sentiment


# Define Stopwords
__covid_19_stopwords = [
    "covid",
    "coronavirus",
    "pandemic",
    "virus",
    "lockdown",
    "quarantine",
    "vaccine",
]
__custom_stopwords = __covid_19_stopwords + stopwords.words()

# Lemmatizer
__lemma = WordNetLemmatizer()


def get_part_of_speech(tag):
    if tag.startswith("J"):
        return wordnet.ADJ

    elif tag.startswith("V"):
        return wordnet.VERB

    elif tag.startswith("N"):
        return wordnet.NOUN

    elif tag.startswith("R"):
        return wordnet.ADV

    else:
        return wordnet.NOUN


def sentence_lemmatizer(sentence: str) -> str:
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    word_lemma = [
        __lemma.lemmatize(word, get_part_of_speech(tag)) for word, tag in pos_tags
    ]
    return " ".join(word_lemma)


def __text_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    # Clean the text
    data["text"] = data["tweet"].str.lower()
    data["text"] = data["text"].apply(contractions.fix)
    data["text"] = data["text"].str.replace(r"https:\W.+", "", regex=True)
    data["text"] = data["text"].str.replace(r"@\w+|&\w+", "", regex=True)
    data["text"] = data["text"].str.replace(
        f"[{re.escape(string.punctuation)}]", " ", regex=True
    )
    data["text"] = data["text"].str.replace(r"\d+\w+", "", regex=True)

    # Sentence lemmatization
    data["text"] = data["text"].apply(sentence_lemmatizer)

    # Handle encoding and decoding issues
    data["text"] = data["text"].apply(lambda s: s.encode("ascii", "ignore"))
    data["text"] = data["text"].apply(lambda s: s.decode("utf-8"))

    # Remove stopwords (and potentially filter short words)
    data["text"] = data["text"].apply(
        lambda text: [
            word
            for word in text.split()
            if word not in __custom_stopwords and len(word) > 2
        ]
    )
    data["text"] = data["text"].str.join(" ")
    return data[["tweet", "text", "sentiment"]]

# Process one-dimensional text (sentence/paragraph)
def text_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """This function performs text preprocessing on a pandas DataFrame.
    Args:
        data (pd.DataFrame): The input DataFrame containing the text data.
    Returns:
        pd.DataFrame: A DataFrame with two features:
            - **tweet:** The original, non-preprocessed text.
            - **text:** The preprocessed text.
     """

    # Clean the text
    data["text"] = data["tweet"].str.lower()
    data["text"] = data["text"].apply(contractions.fix)
    data["text"] = data["text"].str.replace(r"https:\W.+", "", regex=True)
    data["text"] = data["text"].str.replace(r"@\w+|&\w+", "", regex=True)
    data["text"] = data["text"].str.replace(
        f"[{re.escape(string.punctuation)}]", " ", regex=True
    )
    data["text"] = data["text"].str.replace(r"\d+\w+", "", regex=True)

    # Sentence lemmatization
    data["text"] = data["text"].apply(sentence_lemmatizer)

    # Handle encoding and decoding issues
    data["text"] = data["text"].apply(lambda s: s.encode("ascii", "ignore"))
    data["text"] = data["text"].apply(lambda s: s.decode("utf-8"))

    # Remove stopwords (and potentially filter short words)
    data["text"] = data["text"].apply(
        lambda text: [
            word
            for word in text.split()
            if word not in __custom_stopwords and len(word) > 2
        ]
    )
    data["text"] = data["text"].str.join(" ")
    return data[["tweet", "text"]]


def __preprocessing(path: str, data_size=None) -> pd.DataFrame:
    # Loading complete data file
    # ToDo: remove head for full data processing
    # TODO: Uncomment for faster testing
    if data_size is None or data_size == 0:
        data = __reading_file(path)
        # sys.exit("Data Size is not none.")
    else:
        print("Data Size is not none")
        data = __reading_file(path).head(data_size)

    print(data.columns)
    # Preprocess pandas data
    data = data.rename(columns=str.lower)
    data = data[["originaltweet", "sentiment"]]
    data = data.rename(columns={"originaltweet": "tweet"})
    data = data.astype(str)

    # Lowercasing column names
    for c in data.columns:
        data[c] = data[c].str.lower()

    # Reduce label output into three labels i.e. neutral, negative and positive
    data["sentiment"] = data["sentiment"].apply(__consolidate_sentiment)

    # Cleaning and preprocessing tweets/text
    dataframe = __text_preprocessing(data)
    dataframe = (
        dataframe[~(dataframe.text.str.split().apply(lambda s: len(s)) < 2)]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return dataframe


def main() -> None:
    print("Started preprocessing ...")
    # Processed DataSet
    params_loader = load_parameters("configs.yml")

    data_dir = params_loader["data"]
    data_size = data_dir["processed"]["data_size"]

    print(data_size)
    # Command-line args
    raw_path_ = f"{sys.argv[1]}/{data_dir["raw"]["file"]}"
    processed_path_ = f"{sys.argv[2]}/{data_dir["processed"]["file"]}"

    os.makedirs(sys.argv[2], exist_ok=True)

    print("Parent output folder: ", sys.argv[2])

    try:
        # Raise custom exception only if the folder has no csv files.
        if not list(Path(raw_path_).glob("*.csv")):
            raise ValueError(f"There is no `csv` file in `{raw_path_}`.")
        
        df = __preprocessing(Path(raw_path_), data_size)
        df.to_csv(processed_path_, index=False)
        print("Process completed!")
    except ValueError as e:
        print(f"{e}")

if __name__ == "__main__":
    main()
