import os
import warnings
import kagglehub
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import re
import nltk
import string
import contractions
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download("popular", quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


# "https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data"
def download_dataset(URL: str) -> pd.DataFrame:

    training_path = Path(f"{URL}/Corona_NLP_train.csv")
    testing_path = Path(f"{URL}/Corona_NLP_test.csv")

    train_df = pd.read_csv(training_path, encoding="latin1")
    test_df = pd.read_csv(testing_path, encoding="latin1")

    return pd.concat([train_df, test_df], axis=0)


# Download and load the dataset
URL = kagglehub.dataset_download("datatattle/covid-19-nlp-text-classification")
df = download_dataset(URL)

# Preprocess the data
df = df.rename(columns=str.lower)
df = df[["originaltweet", "sentiment"]]
df = df.rename(columns={"originaltweet": "tweet"})
df = df.astype(str)

for c in df.columns:
    df[c] = df[c].str.lower()

df = df.sample(frac=1, random_state=43).reset_index(drop=True)

# Plot sentiment distributions
def plot_sentiment_dist(df: pd.DataFrame, palette="colorblind") -> None:

    sentiment_df = df["sentiment"].value_counts().reset_index(name="size")

    _, ax = plt.subplots(figsize=(10, 4))
    plt.style.use("ggplot")

    sns.barplot(
        sentiment_df, x="sentiment", y="size", hue="sentiment", palette=palette, gap=0.5, ax=ax
    )

    # Axis Labels and Title
    ax.set_title("Consolidated Sentiment Distribution", fontsize=12, alpha=0.8)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("")
    plt.xticks(rotation=45)

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

# All sentiments included
# plot_sentiment_dist(df)

# Consolidate negative and postive sentiments
def consolidate_sentiment(sentiment: str) -> str:
    if sentiment == "extremely positive":
        return "positive"
    elif sentiment == "extremely negative":
        return "negative"
    return sentiment

data = df.copy()

data["sentiment"] = data["sentiment"].apply(consolidate_sentiment)

# consolidated
# plot_sentiment_dist(data, "viridis")

# Define Stopwords
covid_19_stopwords = ["covid", "coronavirus", "pandemic", "virus", "lockdown", "quarantine", "vaccine"]
custom_stopwords = covid_19_stopwords + stopwords.words()

# Lemmatizer
lemma = WordNetLemmatizer()

def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:

    data = df.copy()

    # Clean the text
    data['text'] = data['tweet'].str.lower()
    data['text'] = data['text'].apply(contractions.fix)
    data['text'] = data['text'].str.replace(r'https:\W.+','', regex=True)
    data['text'] = data['text'].str.replace(r'@\w+|&\w+','', regex=True)
    data['text'] = data['text'].str.replace(r'[%s]'%re.escape(string.punctuation),' ', regex=True)
    data['text'] = data['text'].str.replace(r'\d+\w+','', regex=True)

    # Tokenize and preprocess
    data['text'] = data['text'].apply(word_tokenize)
    data['text'] = data['text'].apply(lambda tokens: [lemma.lemmatize(word) for word in tokens])
    data['text'] = data['text'].str.join(" ")

    # Handle encoding and decoding issues
    data['text'] = data['text'].apply(lambda s: s.encode('ascii', 'ignore'))
    data['text'] = data['text'].apply(lambda s: s.decode('utf-8'))

    # Remove stopwords (and potentially filter short words)
    data['text'] = data['text'].apply(lambda text: [word for word in text.split() if word not in custom_stopwords and len(word) > 2])
    data['text'] = data['text'].str.join(" ")
    return data[['tweet', 'text', 'sentiment']]

# Processed DataSet
dataframe = preprocess_text(data)
dataframe = (
    dataframe[~(dataframe.text.str.split()
    .apply(lambda s: len(s)) < 2)]
    .dropna().drop_duplicates()
    .reset_index(drop=True)
)


if __name__ == "__main__":
	# Naming file output
	FILE_NAME = "covid19_tweets.csv"
	FILE_OUTPUT = os.path.join("data/processed", FILE_NAME)

	# Log existing files into file_log.txt
	!find . -maxdepth 2 -type f > file_log.txt

	# Load existing files into a pyton list
	with open("file_log.txt", "r") as f:
	    list_files = [file.split("/")[-1].rstrip() for file in f.readlines()]
	    print(list_files)

	if FILE_NAME not in list_files:
	    print("Saving processed dataset ...")
	    dataframe.to_csv(FILE_OUTPUT, index=False)
	else:
	    print(f"File '{FILE_OUTPUT}' already created.")