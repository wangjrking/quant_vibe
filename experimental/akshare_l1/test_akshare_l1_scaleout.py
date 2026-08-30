from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd

import akshare_l1_experimental as base
import akshare_l1_scaleout as scaleout


class ScaleoutUnitTests(unittest.TestCase):
    def test_retry_recovers_with_exponential_backoff(self) -> None:
        attempts = {"count": 0}
        sleeps: list[float] = []

        def call() -> pd.DataFrame:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ConnectionError("temporary")
            return pd.DataFrame({"value": [1]})

        frame, count = scaleout.retry_call(call, max_retries=2, base_delay_seconds=1, sleep=sleeps.append)
        self.assertEqual(count, 3)
        self.assertEqual(sleeps, [1, 2])
        self.assertEqual(len(frame), 1)

    def test_retry_failure_is_not_completed(self) -> None:
        with self.assertRaises(scaleout.SourceCallFailure) as context:
            scaleout.retry_call(lambda: (_ for _ in ()).throw(TimeoutError("blocked")), max_retries=2, sleep=lambda _: None)
        self.assertEqual(context.exception.attempts, 3)
        self.assertEqual(context.exception.error_type, "TimeoutError")

    def test_empty_response_is_legal_source_empty_input(self) -> None:
        frame, attempts = scaleout.retry_call(lambda: pd.DataFrame(), sleep=lambda _: None)
        self.assertTrue(frame.empty)
        self.assertEqual(attempts, 1)

    def test_ttl_hit_and_expiry(self) -> None:
        current = datetime(2026, 7, 15, 1, 0, tzinfo=timezone(timedelta(hours=8)))
        self.assertTrue(scaleout.cache_fresh((current - timedelta(hours=1)).isoformat(), 7200, current))
        self.assertFalse(scaleout.cache_fresh((current - timedelta(hours=3)).isoformat(), 7200, current))

    def test_call_and_stock_limits(self) -> None:
        self.assertEqual(scaleout.bounded_codes(["688006.SH", "688006.SH"], 1, 1), ["688006.SH"])
        with self.assertRaises(ValueError):
            scaleout.bounded_codes(["000001.SZ", "600000.SH"], 1, 1)
        with self.assertRaises(ValueError):
            scaleout.bounded_codes(["920012.BJ"], 1, 1)

    def test_idempotent_merge(self) -> None:
        current = pd.DataFrame({"id": ["a"], "value": [1]})
        incoming = pd.DataFrame({"id": ["a"], "value": [1]})
        merged = scaleout.merge_deduplicated(current, incoming, ["id"])
        self.assertEqual(len(merged), 1)

    def test_no_bj_gate_filters_entire_materialization(self) -> None:
        frame = pd.DataFrame({"ts_code": ["600000.SH", "920012.BJ"], "value": [1, 2]})
        filtered = scaleout.enforce_no_bj(frame)
        self.assertEqual(filtered["ts_code"].tolist(), ["600000.SH"])

    def test_checkpoint_resume_preserves_completed_days(self) -> None:
        original = scaleout.ANNOUNCEMENT_CHECKPOINT
        try:
            with tempfile.TemporaryDirectory(dir=base.EXPERIMENTAL_ROOT) as temporary:
                scaleout.ANNOUNCEMENT_CHECKPOINT = Path(temporary) / "checkpoint.json"
                checkpoint = scaleout.announcement_checkpoint("20260701", "20260702")
                checkpoint["dates"]["20260701"]["status"] = "completed"
                scaleout.atomic_json(scaleout.ANNOUNCEMENT_CHECKPOINT, checkpoint)
                resumed = scaleout.announcement_checkpoint("20260701", "20260702")
                self.assertEqual(resumed["dates"]["20260701"]["status"], "completed")
                self.assertEqual(resumed["dates"]["20260702"]["status"], "pending")
        finally:
            scaleout.ANNOUNCEMENT_CHECKPOINT = original

    def test_production_path_is_blocked(self) -> None:
        with self.assertRaises(ValueError):
            base.validate_output_root(base.PRODUCTION_ROOT / "forbidden")

    def test_financial_batch_size_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            scaleout.run_financial_batch(object(), [], 0)
        with self.assertRaises(ValueError):
            scaleout.run_financial_batch(object(), [], 501)
        with self.assertRaises(ValueError):
            scaleout.run_financial_batch(object(), [], 1, 0)


class ScaleoutArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(scaleout.MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_scaleout_contract_fields(self) -> None:
        required = {
            "table", "source", "status", "db_path", "source_params", "date_semantics",
            "natural_key", "checkpoint_path", "resume_policy", "sample_scope", "coverage",
            "ttl_seconds", "adjustment",
        }
        for name, asset in self.manifest["assets"].items():
            self.assertEqual(asset["table"], name)
            self.assertTrue(required.issubset(asset))
            self.assertNotIn("production", Path(asset["db_path"]).parts)

    def test_one_table_one_file_no_bj_and_zero_duplicates(self) -> None:
        files = sorted(scaleout.ROOT.glob("*/*.duckdb"))
        self.assertEqual(len(files), 5)
        for name, asset in self.manifest["assets"].items():
            connection = duckdb.connect(asset["db_path"], read_only=True)
            try:
                tables = connection.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main' AND table_type='BASE TABLE'"
                ).fetchall()
                columns = [row[1] for row in connection.execute(f"PRAGMA table_info('{name}')").fetchall()]
                bj = connection.execute(f'SELECT COUNT(*) FROM "{name}" WHERE ts_code LIKE \'%.BJ\'').fetchone()[0] if "ts_code" in columns else 0
            finally:
                connection.close()
            self.assertEqual(tables, [(name,)])
            self.assertEqual(bj, 0)
            self.assertEqual(asset["bj_rows"], 0)
            self.assertTrue(all(value == 0 for value in asset["natural_key_null_counts"].values()))
            self.assertEqual(asset["duplicate_key_groups"], 0)

    def test_minute_cache_is_completed_session_unadjusted(self) -> None:
        asset = self.manifest["assets"]["stock_minutes_1m"]
        self.assertEqual(asset["source"], "AKShare.stock_zh_a_minute")
        self.assertEqual(asset["adjustment"]["semantics"], "unadjusted/raw_price")
        self.assertIn("not real-time", asset["date_semantics"])
        self.assertEqual(asset["sample_scope"]["trade_date"], self.manifest["trade_date_gate"]["latest_completed_trade_date"])

    def test_sentiment_remains_blocked(self) -> None:
        asset = self.manifest["assets"]["social_sentiment_snapshot"]
        self.assertTrue(asset["status"].startswith("blocked_"))
        self.assertTrue(asset["coverage"]["blocked"])


if __name__ == "__main__":
    unittest.main()
