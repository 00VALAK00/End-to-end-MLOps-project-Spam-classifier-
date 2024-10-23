import unittest
import pandas as pd
from steps.feature_engineering import FeatureEngineering


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            {"Message_body":
                [
                    "13,Here is your discount code RP176781. To stop further messages reply stop. [URL] Customer Services [PHONE_NUMBER].",
                    "18,Valentines Day Special! Win over £1000 in our quiz and take your partner on the trip of a lifetime! Send GO to [URL] now. 150p/msg rcvd. CustCare:[PHONE_NUMBER]."

                ]
            })

        self.fe = FeatureEngineering(self.df)
        self.df = self.fe.apply_feature_engineering()
        print(self.df)

    def test_output_shape(self):
        self.assertEqual(self.df.shape[1], 4), f"Expected cols numb to be 4 got {self.df.shape[1]}"

    def test_new_features(self):
        self.assertEqual(self.df["url_count"].tolist(),
                         [1, 1]), f"Expected url_count to be [1,1] got {self.df['url_count'].tolist()}"
        self.assertEqual(self.df["phone_count"].tolist(),
                         [1, 1]), f"Expected phone_number to be [0,1] got {self.df['phone_count'].tolist()}"


if __name__ == '__main__':
    unittest.main()
