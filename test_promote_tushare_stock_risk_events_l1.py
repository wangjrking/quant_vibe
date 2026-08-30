from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from promote_tushare_stock_risk_events_l1 import atomic_json, canonicalize, quality


class PromotionContractTest(unittest.TestCase):
    def test_canonicalize_filters_bj_and_deduplicates(self) -> None:
        frame = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20260821", "reason": "x", "period": "1"},
                {"ts_code": "000001.SZ", "trade_date": "20260821", "reason": "x", "period": "1"},
                {"ts_code": "920001.BJ", "trade_date": "20260821", "reason": "x", "period": "1"},
            ]
        )
        keys = ("ts_code", "trade_date", "reason", "period")
        result = canonicalize(frame, keys, "trade_date")
        self.assertEqual(result["ts_code"].tolist(), ["000001.SZ"])
        self.assertEqual(quality(result, keys, "trade_date")["duplicate_key_groups"], 0)
        self.assertEqual(quality(result, keys, "trade_date")["bj_rows"], 0)

    def test_descriptive_null_is_preserved_but_not_a_required_key_failure(self) -> None:
        frame = pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": "20260821", "reason": None, "period": None}]
        )
        keys = ("ts_code", "trade_date", "reason", "period")
        result = canonicalize(frame, keys, "trade_date")
        result_quality = quality(result, keys, "trade_date")
        self.assertEqual(result_quality["required_key_nulls"], 0)
        self.assertEqual(result_quality["descriptive_key_nulls"], 2)

    def test_absent_prestate_rollback_contract_is_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollback_manifest.json"
            payload = {
                "assets": [
                    {
                        "active_path": "stk_shock.parquet",
                        "pre_publish_state": "absent",
                        "backup_path": None,
                        "backup_sha256": None,
                        "rollback_action": "remove_newly_published_asset",
                    }
                ]
            }
            atomic_json(path, payload)
            self.assertIn('"pre_publish_state": "absent"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
