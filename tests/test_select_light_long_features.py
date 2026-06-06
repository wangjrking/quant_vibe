import unittest

import pandas as pd

from select_light_long_features import prune_correlated_features


class SelectLightLongFeaturesTests(unittest.TestCase):
    def test_prune_correlated_features_keeps_forced_and_diversified(self):
        frame = pd.DataFrame(
            {
                "amount": [1, 2, 3, 4],
                "turnover_rate": [4, 3, 2, 1],
                "dup_amount": [1, 2, 3, 4],
                "independent": [1, 1, 2, 3],
            }
        )

        selected = prune_correlated_features(
            frame,
            ranked_features=["dup_amount", "independent"],
            force_include=["amount", "turnover_rate"],
            max_abs_corr=0.95,
            top_n=3,
        )

        self.assertEqual(selected[:2], ["amount", "turnover_rate"])
        self.assertIn("independent", selected)
        self.assertNotIn("dup_amount", selected)


if __name__ == "__main__":
    unittest.main()
