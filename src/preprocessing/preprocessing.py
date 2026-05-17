import argparse
import re
import unicodedata
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer


CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "wouldn't": "would not",
    "wasn't": "was not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "aren't": "are not",
    "weren't": "were not",
    "i'm": "i am",
    "we're": "we are",
    "they're": "they are",
    "you're": "you are",
    "we've": "we have",
    "i've": "i have",
    "you've": "you have",
    "they've": "they have",
    "i'd": "i would",
    "who'd": "who would",
    "it's": "it is",
    "he's": "he is",
    "that's": "that is",
    "there's": "there is",
    "who's": "who is",
    "what's": "what is",
    "let's": "let us",
}

NO_APOSTROPHE_CONTRACTIONS = {
    "weve": "we have",
    "ive": "i have",
    "youve": "you have",
    "theyve": "they have",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "cant": "can not",
    "wont": "will not",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "hasnt": "has not",
    "havent": "have not",
    "wouldnt": "would not",
    "couldnt": "could not",
    "shouldnt": "should not",
    "hes": "he is",
    "shes": "she is",
    "thats": "that is",
    "theres": "there is",
    "whos": "who is",
    "whats": "what is",
}

SPECIAL_NORMALIZATIONS = {
    "obamas": "obama",
    "bushs": "bush",
    "mccains": "mccain",
    "clintons": "clinton",
}

NEGATIONS = {"no", "not", "nor", "never"}
TOP_PARTIES = {"republican", "democrat", "none", "organization", "independent"}

lemmatizer = WordNetLemmatizer()


def download_nltk_resources() -> None:
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for resource in resources:
        nltk.download(resource, quiet=True)


def clean_category(series: pd.Series, missing_value: str = "missing") -> pd.Series:
    return series.fillna(missing_value).astype(str).str.lower().str.strip()


def keep_top_categories(series: pd.Series, top_n: int, other_label: str = "other") -> pd.Series:
    top_values = series.value_counts().head(top_n).index
    return series.apply(lambda value: value if value in top_values else other_label)


def expand_contractions(text: str) -> str:
    text = str(text).lower().replace("’", "'")

    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)

    for contraction, expanded in NO_APOSTROPHE_CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)

    for original, normalized in SPECIAL_NORMALIZATIONS.items():
        text = re.sub(r"\b" + re.escape(original) + r"\b", normalized, text)

    # Possessives with apostrophe: obama's -> obama.
    text = re.sub(r"\b([a-z]+)'s\b", r"\1", text)
    return text


def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\bNo\.\s*(\d+)", r"number \1", text, flags=re.IGNORECASE)
    text = text.replace("’", "'")
    text = expand_contractions(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8", "ignore")
    text = re.sub(r"\S+\.(?:edu|com|gov|net)\S*", "", text)
    return text.lower()


def get_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_statement(text: str, stop_words: set[str]) -> str:
    text = normalize_text(text)
    text = re.sub(r"\d+/\d+(?:st|nd|rd|th)?", " fraction ", text)
    text = re.sub(r"\d+(?:st|nd|rd|th)", " ordinal ", text)
    text = re.sub(r"\d+(?:[.,]\d+)*", " number ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() or word in NEGATIONS]
    tokens = [word for word in tokens if len(word) >= 2]
    tokens = [word for word in tokens if word not in stop_words]

    pos_tags = nltk.pos_tag(tokens)
    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
    ]
    tokens = [word for word in tokens if len(word) >= 2]

    return " ".join(tokens)


def add_subject_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df["subject"] = clean_category(df["subject"])
    df["subject_list"] = df["subject"].str.split(",")
    df["subject_list"] = df["subject_list"].apply(lambda values: [value.strip() for value in values])

    mlb = MultiLabelBinarizer()
    subject_encoded = mlb.fit_transform(df["subject_list"])
    subject_cols = [f"subject_{subject}" for subject in mlb.classes_]

    subject_df = pd.DataFrame(subject_encoded, columns=subject_cols, index=df.index)
    df = pd.concat([df, subject_df], axis=1)

    return df, subject_cols


def preprocess_dataframe(
    df: pd.DataFrame,
    speaker_top_n: int = 50,
    speaker_job_top_n: int = 50,
    state_top_n: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["speaker_job_missing"] = df["speaker_job"].isnull().astype(int)
    df["state_info_missing"] = df["state_info"].isnull().astype(int)

    df["speaker"] = clean_category(df["speaker"])
    df["speaker_job"] = clean_category(df["speaker_job"])
    df["state_info"] = clean_category(df["state_info"])
    df["party_affiliation"] = clean_category(df["party_affiliation"])

    stop_words = set(stopwords.words("english")) - NEGATIONS
    df["statement_normalized"] = df["statement"].apply(normalize_text)
    df["statement_clean"] = df["statement"].apply(
        lambda text: preprocess_statement(text, stop_words)
    )

    df, subject_cols = add_subject_features(df)

    df["speaker_top"] = keep_top_categories(df["speaker"], top_n=speaker_top_n)
    df["speaker_job_top"] = keep_top_categories(df["speaker_job"], top_n=speaker_job_top_n)
    df["state_info_top"] = keep_top_categories(df["state_info"], top_n=state_top_n)
    df["party_top"] = df["party_affiliation"].apply(
        lambda value: value if value in TOP_PARTIES else "other"
    )

    return df, subject_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess political claims dataset.")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV path.")
    parser.add_argument("--output", default="data/train_preprocessed.csv", help="Output CSV path.")
    parser.add_argument("--speaker-top-n", type=int, default=50)
    parser.add_argument("--speaker-job-top-n", type=int, default=50)
    parser.add_argument("--state-top-n", type=int, default=30)
    args = parser.parse_args()

    download_nltk_resources()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    df_processed, subject_cols = preprocess_dataframe(
        df,
        speaker_top_n=args.speaker_top_n,
        speaker_job_top_n=args.speaker_job_top_n,
        state_top_n=args.state_top_n,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)

    print(f"Input shape: {df.shape}")
    print(f"Output shape: {df_processed.shape}")
    print(f"Subject columns: {len(subject_cols)}")
    print(f"Saved: {output_path}")
    print("\nColumns to use in model:")
    print(
        [
            "statement_clean",
            "speaker_top",
            "speaker_job_top",
            "state_info_top",
            "party_top",
            "speaker_job_missing",
            "state_info_missing",
        ]
        + subject_cols
    )


if __name__ == "__main__":
    main()
