import unittest

from steps.ingest_data import DataIngestionForTraining
from steps.validate_data import DataValidationForTraining


class TestDataValidation(unittest.TestCase):

    def setUp(self):
        self.train_data, self.val_data = DataIngestionForTraining().ingest_data()

    def test_data_validation(self):
        validated = DataValidationForTraining().validate(self.train_data) & DataValidationForTraining().validate(
            self.val_data)
        self.assertEqual(validated, True)


if __name__ == '__main__':
    unittest.main()
