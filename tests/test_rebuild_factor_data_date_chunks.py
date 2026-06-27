import unittest

import pandas as pd

from rebuild_factor_data_date_chunks import add_cross_sectional_standard_features, build_date_chunks


class RebuildFactorDataDateChunksTests(unittest.TestCase):
    def test_build_date_chunks_adds_past_and_future_overlap(self):
        dates = [f"202601{i:02d}" for i in range(1, 11)]

        chunks = build_date_chunks(dates, chunk_days=3, past_overlap_days=2, future_overlap_days=1)

        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].output_dates, ["20260101", "20260102", "20260103"])
        self.assertEqual(chunks[0].read_start, "20260101")
        self.assertEqual(chunks[0].read_end, "20260104")
        self.assertEqual(chunks[1].output_dates, ["20260104", "20260105", "20260106"])
        self.assertEqual(chunks[1].read_start, "20260102")
        self.assertEqual(chunks[1].read_end, "20260107")

    def test_build_date_chunks_applies_start_and_end_date(self):
        dates = [f"202601{i:02d}" for i in range(1, 11)]

        chunks = build_date_chunks(
            dates,
            chunk_days=4,
            past_overlap_days=1,
            future_overlap_days=2,
            start_date="20260103",
            end_date="20260108",
        )

        self.assertEqual([date for chunk in chunks for date in chunk.output_dates], [
            "20260103",
            "20260104",
            "20260105",
            "20260106",
            "20260107",
            "20260108",
        ])
        self.assertEqual(chunks[0].read_start, "20260102")
        self.assertEqual(chunks[-1].read_end, "20260110")

    def test_add_cross_sectional_standard_features_ranks_by_trade_date(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20260102", "20260102", "20260103", "20260103"],
                "stock_code": ["A", "B", "A", "B"],
                "factor": [1.0, 3.0, 10.0, 20.0],
            }
        )

        out = add_cross_sectional_standard_features(frame, ["factor"], add_rank=True, add_zscore=True)

        self.assertEqual(out["factor_rank"].tolist(), [0.5, 1.0, 0.5, 1.0])
        self.assertAlmostEqual(out.loc[0, "factor_zscore"], -0.70710678, places=6)
        self.assertAlmostEqual(out.loc[1, "factor_zscore"], 0.70710678, places=6)


if __name__ == "__main__":
    unittest.main()
