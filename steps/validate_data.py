import pandas as pd
from config.config_reader import ConfigReader
from config.config_entities import DataValidationConfig
from typing_extensions import Annotated
from zenml.logger import get_logger
from zenml import step
from abc import ABC, abstractmethod

logger = get_logger(__file__)


class DataValidation(ABC):
    def __init__(self):
        self.config: DataValidationConfig = ConfigReader().import_data_validation_config()

    @abstractmethod
    def validate(self, df: pd.DataFrame):
        pass


class DataValidationForTraining(DataValidation):
    def __init__(self):
        super().__init__()

    def validate(self, df: pd.DataFrame) -> Annotated[bool, "validated"]:

        validated = None
        schema: dict = self.config.schema
        labels_values: list = self.config.label_values

        if df.shape[1] != 2:
            validated = False
            logger.error(f"Data Validation failed due to an unexpected number of columns {df.columns.tolist()}")
            return validated

        for col_name in df.columns.tolist():
            # Check columns names incompatibility
            if col_name not in schema.keys():
                validated = False
                logger.error(f"Data Validation failed due to an unexpected column {col_name}")

                # verify data types
            elif df[col_name].dtype != schema[col_name]:
                try:
                    # cast it as string data type
                    df[col_name] = df[col_name].astype(schema[col_name])
                except TypeError as e:
                    logger.error(f"Data Validation failed due to an unexpected column {col_name}")
                    validated = False
        # validate label values

        assert set(df["Label"].unique().tolist()) == set(labels_values)
        validated = True

        return validated


def validate_data():
    pass
