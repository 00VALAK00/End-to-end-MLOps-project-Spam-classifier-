import pandas as pd
from string import punctuation
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how='any')


def general_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Eliminate spaces and punctuations
    df["Message_body"] = df["Message_body"].str.lower()
    df["Message_body"] = df["Message_body"].str.replace(to_replace="\s+",
                                                        regex=True,
                                                        value=" ")

    df["Message_body"] = df["Message_body"].str.replace(to_replace=punctuation,
                                                        value="",
                                                        regex=True
                                                        )

    return df


def apply_stem(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize the stemmer
    lemmatizer = WordNetLemmatizer()

    # Define a function to apply stemming
    def stem(text: str) -> str:
        # Tokenize the text
        words = word_tokenize(text)
        # Apply stemming to each word
        lemmas = [lemmatizer.lemmatize(word) for word in words]
        # Join the stemmed words back into a string
        return " ".join(lemmas)

    # Apply the stem function to each entry in the Series
    df['Message_body'] = df['Message_body'].apply(stem)

    return df


def mask_urls_and_phone_numbers(df: pd.DataFrame) -> pd.DataFrame:
    # replace urls and numbers by [URL] & [PHONE_NUMBER] tokens
    url_pattern = r'http\S+'
    phone_numbers_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10}\b'

    df["Message_body"] = df["Message_body"].str.replace(url_pattern,
                                                        "[URL]",
                                                        regex=True)
    df["Message_body"] = df["Message_body"].str.replace(url_pattern,
                                                        "[PHONE_NUMBER]",
                                                        regex=True)
    return df


def pre_feature_engineering_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = handle_missing_values(df)
    df = general_cleaning(df)
    df = mask_urls_and_phone_numbers(df)
    df = apply_stem(df)

    return df
