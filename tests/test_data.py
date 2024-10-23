import unittest

from steps.ingest_data import DataIngestionForTraining, DataIngestionForInference


class TrainDataIngestion(unittest.TestCase):
    def setUp(self):
        train, val = DataIngestionForTraining().ingest_data()
        self.train_data = train
        self.val_data = val

    def test_data_consistency(self):
        # test number of cols
        self.assertEqual(self.train_data.shape[1], self.val_data.shape[1])

        self.assertFalse(any([self.train_data.empty, self.val_data.empty]))

        expected_columns = ["Message_body", "Label"]
        self.assertListEqual(list(self.val_data.columns), expected_columns)
        self.assertListEqual(list(self.train_data.columns), expected_columns)


if __name__ == '__main__':
    unittest.main()
