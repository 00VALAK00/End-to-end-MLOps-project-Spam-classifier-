import os

from config.config_entities import *
from utils.common import read_yaml

config_path = Path(__file__).parent / 'config.yml'


class ConfigReader:
    def __init__(self, config_file: Path = config_path):
        self.config_file = config_file
        self.data_status = config_file
        self.config_file = read_yaml(self.config_file)

    def import_data_ingestion_config(self) -> DataIngestionConfig:
        config: dict = self.config_file.get("data_ingestion")
        data_ingestion_config = DataIngestionConfig(
            data_path=config.get("data_path"),
            train_data_path=config.get("train_data_path"),
            val_data_path=config.get("val_data_path")
        )
        return data_ingestion_config

    def import_data_validation_config(self) -> DataValidationConfig:
        config: dict = self.config_file.get("data_validation")
        data_validation_config = DataValidationConfig(
            schema=config.get("schema"),
            label_values=config.get("label_values")
        )
        return data_validation_config
