
import unittest
import pandas as pd
import data

class TestDataUtils(unittest.TestCase):
    def test_get_fold_queries(self):
        """
        Test that it can sum a list of integers
        """
        test_data = {
            "Q_name": ["test1", "test2", "test3", "test4"],
            "split": ["train", "train", "test", "test"],
            "fold": [1, 2, 1, 2]
        }
        test_df = pd.DataFrame(test_data)
        res = data.get_fold_queries(1, split=None, df=test_df)
        self.assertEqual(len(res), 2)
        self.assertListEqual(res, ['test1','test3'])

        res = data.get_fold_queries(1, split='train', df=test_df)
        self.assertEqual(len(res), 1)
        self.assertListEqual(res, ['test1'])



if __name__ == '__main__':
    unittest.main()
