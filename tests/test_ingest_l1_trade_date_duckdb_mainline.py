from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from ingest_l1_trade_date_duckdb_mainline import (
    build_l1_handoff_contract,
    delete_duckdb_target,
    duplicate_rows,
    normalize_codes,
    update_duckdb_target,
    validate_sources,
    verify_preexisting_sparse_event_slice,
)
from workflow_contract import validate_layer_handoff_contract


class L1TradeDateIngestTests(unittest.TestCase):
    def test_normalize_codes_excludes_bj(self) -> None:
        frame = pd.DataFrame({"ts_code": ["000001.SZ", "600000.SH", "920001.BJ"]})
        self.assertEqual(normalize_codes(frame), {"000001.SZ", "600000.SH"})

    def test_duplicate_rows_uses_natural_key(self) -> None:
        frame = pd.DataFrame({"ts_code": ["000001.SZ", "000001.SZ"], "trade_date": ["20260720", "20260720"]})
        self.assertEqual(duplicate_rows(frame, ("ts_code", "trade_date")), 2)

    def test_source_validation_requires_driver_alignment(self) -> None:
        date = "20260720"
        daily = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [date]})
        frames = {
            "daily_data": daily,
            "daily_index_data": daily.copy(),
            "stk_factor": daily.copy(),
            "moneyflow": daily.copy(),
            "cyq_perf": pd.DataFrame({"ts_code": [], "trade_date": []}),
            "limit_list_data": daily.copy(),
            "adj_factor": daily.copy(),
            "stock_st": daily.copy(),
            "index_daily": pd.DataFrame({"ts_code": ["000300.SH", "000905.SH", "000852.SH", "932000.CSI"], "trade_date": [date] * 4}),
            "top_list": daily.copy(),
        }
        source_names = {
            "daily_data": "daily",
            "daily_index_data": "daily_basic",
            "limit_list_data": "limit_list",
        }
        metrics = {
            table: {"source_table": source_names.get(table, table)}
            for table in frames
        }
        result = validate_sources(frames, metrics, date)
        self.assertFalse(result["passed"])
        self.assertIn("cyq_perf stock domain differs from daily_data", result["blockers"])

    def test_source_validation_accepts_documented_single_code_gap(self) -> None:
        date = "20260721"
        daily = pd.DataFrame(
            {"ts_code": ["000001.SZ", "688806.SH"], "trade_date": [date, date]}
        )
        complete = daily.copy()
        frames = {
            "daily_data": daily,
            "daily_index_data": complete.copy(),
            "stk_factor": complete.iloc[:1].copy(),
            "moneyflow": complete.copy(),
            "cyq_perf": complete.copy(),
            "limit_list_data": complete.copy(),
            "adj_factor": complete.copy(),
            "stock_st": complete.copy(),
            "index_daily": pd.DataFrame(
                {
                    "ts_code": ["000300.SH", "000905.SH", "000852.SH", "932000.CSI"],
                    "trade_date": [date] * 4,
                }
            ),
            "top_list": complete.copy(),
        }
        source_names = {
            "daily_data": "daily",
            "daily_index_data": "daily_basic",
            "limit_list_data": "limit_list",
        }
        metrics = {
            table: {"source_table": source_names.get(table, table)}
            for table in frames
        }
        result = validate_sources(
            frames,
            metrics,
            date,
            source_limited_codes={"stk_factor": {"688806.SH"}},
        )
        self.assertTrue(result["passed"])
        self.assertIn(
            "stk_factor has documented source-limited codes: ['688806.SH']",
            result["warnings"],
        )

    def test_duckdb_target_update_and_delete_are_transactional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "daily_data.duckdb"
            connection = duckdb.connect(str(path))
            try:
                connection.execute("CREATE TABLE daily_data(ts_code VARCHAR, trade_date VARCHAR, close DOUBLE)")
                connection.execute("INSERT INTO daily_data VALUES ('000001.SZ','20260717',10.0)")
            finally:
                connection.close()
            incoming = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260720"], "close": [11.0]})
            update_duckdb_target(path, "daily_data", incoming, "20260720")
            connection = duckdb.connect(str(path), read_only=True)
            try:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0], 2)
            finally:
                connection.close()
            delete_duckdb_target(path, "daily_data", "20260720")
            connection = duckdb.connect(str(path), read_only=True)
            try:
                self.assertEqual(connection.execute("SELECT MAX(trade_date) FROM daily_data").fetchone()[0], "20260717")
            finally:
                connection.close()

    def test_preexisting_sparse_event_slice_requires_exact_three_face_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parquet_path = root / "stk_shock.parquet"
            db_path = root / "stk_shock.duckdb"
            source = pd.DataFrame(
                {
                    "trade_date": ["20260821"],
                    "ts_code": ["000001.SZ"],
                    "reason": ["sample"],
                }
            )
            source.to_parquet(parquet_path, index=False)
            connection = duckdb.connect(str(db_path))
            try:
                connection.register("incoming", source)
                connection.execute("CREATE TABLE stk_shock AS SELECT * FROM incoming")
            finally:
                connection.close()

            result = verify_preexisting_sparse_event_slice(
                source,
                parquet_path,
                db_path,
                "stk_shock",
                ("trade_date", "ts_code", "reason"),
                "20260821",
                "trade_date",
            )
            self.assertEqual(result["status"], "preexisting_verified_reuse")
            self.assertEqual(result["write_action"], "unchanged")

            changed = source.copy()
            changed.loc[0, "reason"] = "different"
            with self.assertRaisesRegex(RuntimeError, "value mismatch"):
                verify_preexisting_sparse_event_slice(
                    changed,
                    parquet_path,
                    db_path,
                    "stk_shock",
                    ("trade_date", "ts_code", "reason"),
                    "20260821",
                    "trade_date",
                )
    def test_generated_handoff_uses_standard_contract_schema(self) -> None:
        trade_date = "20260722"
        report = {
            "official_sdk_only": True,
            "tables_not_landed": [],
            "source_limited": [],
            "results": {
                "daily_data": {
                    "passed": True,
                    "duckdb": {
                        "path": "D:/assets/daily_data.duckdb",
                        "max_trade_date": trade_date,
                        "duplicate_groups_all": 0,
                        "bj_rows_all": 0,
                        "one_table_one_file": True,
                    },
                }
            },
        }
        contract = build_l1_handoff_contract(
            report,
            workflow_id="incremental-trading-signal-20260722",
            trade_date=trade_date,
            evidence_paths=["D:/reports/l1_complete.json"],
        )
        self.assertEqual(
            validate_layer_handoff_contract(
                contract,
                expected_layer="L1",
                expected_owner_agent="data-ingestion-agent",
            ),
            [],
        )
        self.assertIn("explicit-qfq", contract["hard_rules"])
        self.assertFalse(contract["allow_next_layer_continue"])


if __name__ == "__main__":
    unittest.main()
