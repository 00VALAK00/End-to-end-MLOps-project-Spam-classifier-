import unittest
from steps.ingest_data import DataIngestionForTraining, DataIngestionForInference
from utils.cleaning_utils import *


class MyTestCase(unittest.TestCase):
    def setUp(self):
        train, val = DataIngestionForTraining().ingest_data()
        self.train_data = train
        self.val_data = val


    def test_general_cleaning(self):
        self.as


if __name__ == '__main__':
    unittest.main()
