import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt_tab')


class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_url_count_columns(self):
        self.df["url_count"] = self.df["Message_body"].map(lambda x: x.count('[URL]'))
        return self.df

    def create_phone_number_count_column(self):
        self.df["phone_count"] = self.df["Message_body"].map(lambda x: x.count('[PHONE_NUMBER]'))
        return self.df

    def create_email_length_column(self):
        self.df["email_length"] = self.df["Message_body"].map(lambda x: len(word_tokenize(x)))
        return self.df

    def apply_feature_engineering(self):
        self.create_url_count_columns()
        self.create_phone_number_count_column()
        self.create_email_length_column()
        return self.df


