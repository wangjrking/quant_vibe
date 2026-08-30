import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "archive"
    / "root-scripts"
    / "research_front_combo_reaudit_20260720.py"
)
SPEC = importlib.util.spec_from_file_location("research_front_combo_reaudit_20260720", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


class ResearchFrontComboReaudit20260720Tests(unittest.TestCase):
    def test_selection_sort_key_ignores_holdout_and_recent(self) -> None:
        left = {
            "train_objective": 1.0,
            "train_top3": 0.2,
            "train_top5": 0.1,
            "train_rank_ic": 0.05,
            "holdout_top5": -9.0,
            "recent63_top5": -9.0,
            "recent20_top5": -9.0,
        }
        right = {
            "train_objective": 1.0,
            "train_top3": 0.2,
            "train_top5": 0.1,
            "train_rank_ic": 0.05,
            "holdout_top5": 9.0,
            "recent63_top5": 9.0,
            "recent20_top5": 9.0,
        }
        self.assertEqual(mod.selection_sort_key(left), mod.selection_sort_key(right))

    def test_validate_required_columns_fail_closed_on_null(self) -> None:
        frame = pd.DataFrame({"a": [1.0, None], "b": [2.0, 3.0]})
        with self.assertRaisesRegex(ValueError, "fail_closed_required_inputs"):
            mod.validate_required_columns(frame, ["a", "b"], "unit_test")

    def test_validate_required_columns_fail_closed_on_absent_column(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        with self.assertRaisesRegex(ValueError, "fail_closed_required_inputs"):
            mod.validate_required_columns(frame, ["a", "missing_col"], "unit_test")

    def test_build_train_pool_uses_series_mask_and_fit_end(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date": ["20251231", "20260102", "20251215"],
                "stock_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "pred_5d_rank": [0.95, 0.99, 0.89],
                "label_5d": [0.2, 0.3, 0.1],
            }
        )
        out = mod.build_train_pool(frame, "5d", target_top_k=5, pool_threshold=0.90)
        self.assertEqual(out["stock_code"].tolist(), ["000001.SZ"])
        self.assertIn("target_top5", out.columns)


if __name__ == "__main__":
    unittest.main()
