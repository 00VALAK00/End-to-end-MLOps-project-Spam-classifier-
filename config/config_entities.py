from dataclasses import dataclass,field
from pathlib import Path
from typing import Dict,List


@dataclass(frozen=True)
class DataIngestionConfig:
    data_path: Path
    train_data_path: Path
    val_data_path: Path


@dataclass(frozen=True)
class DataValidationConfig:
    schema: Dict[str, str] = field(default_factory=lambda: {
        "Message_body": str,
        "Labels": str
    })

    label_values: List[str] = field(default_factory=lambda: ["Spam", "Non-Spam"])


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: str
    data_path: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: str
    train_data: Path
    val_data: Path


@dataclass(frozen=True)
class ModelValidationConfig:
    root_dir: str
    val_data: Path
    test_data: Path
