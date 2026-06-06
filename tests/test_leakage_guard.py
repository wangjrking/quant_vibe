import unittest

from leakage_guard import find_leaky_features, validate_no_leakage


class LeakageGuardTests(unittest.TestCase):
    def test_flags_future_price_and_return_features(self):
        features = [
            "close_rate",
            "pre_yield_rate",
            "post_open",
            "post2_high",
            "10d_yield_rate_rank",
            "open6_yield_rate_rank",
            "2d_tag",
            "gtja_alpha188",
        ]

        leaky = find_leaky_features(features, label="10d_yield_rate")

        self.assertEqual(
            leaky,
            ["post_open", "post2_high", "10d_yield_rate_rank", "open6_yield_rate_rank", "2d_tag"],
        )

    def test_validate_raises_with_clear_message(self):
        with self.assertRaisesRegex(ValueError, "Future-looking features"):
            validate_no_leakage(["close_rate", "post_close"], label="10d_yield_rate")

    def test_allows_current_and_lagged_features(self):
        validate_no_leakage(
            ["close_rate", "pre_yield_rate", "gtja_alpha001", "industry_encode"],
            label="10d_yield_rate",
        )


if __name__ == "__main__":
    unittest.main()
