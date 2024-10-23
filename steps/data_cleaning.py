from utils.cleaning_utils import *


class DataCleaning:
    def __init__(self, handle_url_numbers) -> None:
        self.handle = handle_url_numbers

    def pre_feature_engineering_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = handle_missing_values(df)
        df = general_cleaning(df)
        df = handle_urls_and_phone_numbers(df, self.handle)
        df = apply_stem(df)

        return df
