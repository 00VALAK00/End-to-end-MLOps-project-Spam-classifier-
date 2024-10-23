import pandas as pd
from typing import List,Literal
from pathlib import Path
import sys
parent_dir = Path(__file__).parent
from zenml.logger import
from utils.preprocessing_utils import pre_feature_engineering_preprocessing,post_feature_engineering_preprocessing



class DataPreprocessor:
    def __init__(self,mode=Literal["train","inference"]) -> None:
        self.mode = mode

    def preprocess(self,df:pd.DataFrame) -> pd.DataFrame:
        if self.mode == "inference":
