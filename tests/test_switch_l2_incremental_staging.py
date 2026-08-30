from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "switch_l2_incremental_staging.py"
SPEC = importlib.util.spec_from_file_location("switch_l2_incremental_staging", TOOL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_asset(path: Path, dates: list[str]) -> None:
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "CREATE TABLE STOCK_DAILY_DATA(stock_code VARCHAR, trade_date VARCHAR)"
        )
        conn.executemany(
            "INSERT INTO STOCK_DAILY_DATA VALUES (?, ?)",
            [("000001.SZ", trade_date) for trade_date in dates],
        )
        conn.execute("CHECKPOINT")


class SnapshotBeforeReplaceTest(unittest.TestCase):
    def test_switch_creates_snapshot_before_replacing_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.duckdb"
            staging = root / "staging.duckdb"
            snapshot = root / "snapshot.duckdb"
            validation = root / "validation.json"
            output = root / "switch.json"
            create_asset(active, ["20260812"])
            create_asset(staging, ["20260812", "20260813"])
            expected = sha256(active)
            validation.write_text('{"status":"passed"}', encoding="utf-8")

            argv = [
                str(TOOL_PATH),
                "--active", str(active),
                "--staging", str(staging),
                "--snapshot", str(snapshot),
                "--validation-json", str(validation),
                "--target-trade-date", "20260813",
                "--expected-active-sha256", expected,
                "--output-json", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                MODULE.main()

            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(snapshot.exists())
            self.assertEqual(sha256(snapshot), expected)
            self.assertEqual(result["snapshot"]["sha256"], expected)
            self.assertEqual(result["after"]["metrics"]["max_trade_date"], "20260813")

    def test_existing_snapshot_fails_before_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.duckdb"
            staging = root / "staging.duckdb"
            snapshot = root / "snapshot.duckdb"
            validation = root / "validation.json"
            output = root / "switch.json"
            create_asset(active, ["20260812"])
            create_asset(staging, ["20260813"])
            create_asset(snapshot, ["20260811"])
            expected = sha256(active)
            validation.write_text('{"status":"passed"}', encoding="utf-8")

            argv = [
                str(TOOL_PATH),
                "--active", str(active),
                "--staging", str(staging),
                "--snapshot", str(snapshot),
                "--validation-json", str(validation),
                "--target-trade-date", "20260813",
                "--expected-active-sha256", expected,
                "--output-json", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(RuntimeError, "snapshot path already exists"):
                    MODULE.main()
            self.assertEqual(sha256(active), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
