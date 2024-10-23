from abc import ABC, abstractmethod
from typing_extensions import Annotated
from zenml.logger import get_logger
from zenml import step
from typing import Tuple, Literal
import pandas as pd
from typing import List
from config.config_entities import DataIngestionConfig
from config.config_reader import ConfigReader
from utils.common import read_csv

logger = get_logger(__name__)


class DataIngestion(ABC):
    def __init__(self):
        self.mode: Literal["inference", "train"]
        self.config: DataIngestionConfig = ConfigReader().import_data_ingestion_config()

    @abstractmethod
    def ingest_data(self):
        pass


class DataIngestionForTraining(DataIngestion):
    def __init__(self):
        super().__init__()
        logger.info("mode set to training")

    def ingest_data(self) -> Tuple[
        Annotated[pd.DataFrame, "train_df"],
        Annotated[pd.DataFrame, "val_df"]
    ]:
        train_df = read_csv(self.config.train_data_path)
        val_df = read_csv(self.config.val_data_path)
        return train_df, val_df


class DataIngestionForInference(DataIngestion):
    def __init__(self):
        super().__init__()
        logger.info("mode set to inference")

    def ingest_data(self) -> Annotated[pd.DataFrame, "inference_data"]:
        inference_df = read_csv(self.config.train_data_path)
        inference_df.drop(columns=['Label'], inplace=True)
        return inference_df.sample(round(len(inference_df) * 0.2), random_state=42)


def ingest_data(mode: Literal["inference", "train"] = "train"):
    """Data ingestion step

    Args:
    inference: training or inference mode

    Returns:
    based on the mode returns the data suited for the task
    """

    try:

        logger.info("Started the data ingestion process")
        if mode == "inference":
            inference_df = DataIngestionForInference().ingest_data()
            logger.info("data ingestion process completed")
            return inference_df

        else:
            train_df, val_df = DataIngestionForTraining().ingest_data()
            logger.info("data ingestion process completed")
            return train_df, val_df

    except Exception as e:
        logger.error(f"An error occurred while ingesting data {e}")

