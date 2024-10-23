import yaml
import pandas as pd
from pathlib import Path


def read_yaml(path: Path):
    try:
        print(f"path is : {path}")
        with open(path) as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise e


def read_csv(path: Path, delimiter=",") -> pd.DataFrame:
    return pd.read_csv(path, delimiter=delimiter,
                       encoding="ISO-8859-1",
                       usecols=["Message_body", "Label"],
                       low_memory=False)
