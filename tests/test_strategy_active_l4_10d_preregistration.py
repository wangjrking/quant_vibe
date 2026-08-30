from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

import research_active_l4_10d_market_gated_sleeves_v74_20260722 as v74
import research_active_l4_robust_objective_v56_20260721 as robust


class PreregisteredTenDayTimingTest(unittest.TestCase):
    def test_market_gate_at_t_does_not_read_t_plus_one_open(self):
        opens = np.asarray(
            [
                [10.0, 20.0],
                [10.1, 20.2],
                [10.2, 20.4],
                [10.3, 20.6],
                [10.4, 20.8],
                [10.5, 21.0],
            ],
            dtype=np.float32,
        )
        arrays = {
            "buy_open": opens.copy(),
            "signal_clean": np.ones_like(opens, dtype=np.bool_),
        }
        _, original_gate = v74.apply_market_gate(arrays, lookback=3, threshold=0.0)

        changed = {key: value.copy() for key, value in arrays.items()}
        signal_index = 4
        changed["buy_open"][signal_index] = np.asarray([99.0, 199.0], dtype=np.float32)
        _, changed_gate = v74.apply_market_gate(changed, lookback=3, threshold=0.0)

        self.assertEqual(bool(original_gate[signal_index]), bool(changed_gate[signal_index]))

    def test_observation_truncation_excludes_validation_dates(self):
        arrays = {
            "dates": np.asarray(["20251230", "20251231", "20260102"], dtype="U8"),
            "rank_10d": np.ones((3, 2), dtype=np.float32),
            "stocks": np.asarray(["000001.SZ", "600000.SH"], dtype="U9"),
        }
        result = robust.truncate_observation(arrays, "20251231")

        self.assertEqual(result["dates"].tolist(), ["20251230", "20251231"])
        self.assertEqual(result["rank_10d"].shape, (2, 2))
        self.assertEqual(result["stocks"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
