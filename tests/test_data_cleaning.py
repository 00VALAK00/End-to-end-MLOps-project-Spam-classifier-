import unittest
from steps.data_cleaning import DataCleaning
import pandas as pd


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"Message_body": [
                "13,Here is your discount code RP176781. To stop further messages reply stop. www.regalportfolio.co.uk. Customer Services 08717205546.",
                "18,Valentines Day Special! Win over £1000 in our quiz and take your partner on the trip of a lifetime! Send GO to 83600 now. 150p/msg rcvd. CustCare:08718720201."

            ]
            })

        self.DC = DataCleaning(handle_url_numbers="mask")

    def test_deletion(self):
        # Apply the cleaning function
        df_cleaned = self.DC.pre_feature_engineering_cleaning(self.df)

        # Check if URLs, phone numbers, and numeric codes are masked/removed
        for message in df_cleaned["Message_body"]:
            self.assertNotRegex(message, r'http\S+|www\S+', "URL not removed")
            self.assertNotRegex(message, r'\d{5,}', "Phone numbers or numeric codes not removed")


if __name__ == '__main__':
    unittest.main()
