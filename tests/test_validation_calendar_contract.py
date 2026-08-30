import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from validation_calendar_contract import (
    CalendarContractError,
    audit_asset_calendars,
    audit_sorted_key_rows,
    build_t1_map,
    compare_sorted_key_rows,
)
from tools.validate_validation_calendar_contract import run


class ValidationCalendarContractTests(unittest.TestCase):
    def test_exact_calendar_is_ready(self):
        result = audit_asset_calendars(
            ["20260105", "20260106", "20260107"],
            {
                "l2": ["20260105", "20260106", "20260107"],
                "l4_10d": ["20260105", "20260106", "20260107"],
            },
            start="20260105",
            end="20260107",
            minimum_dates=3,
        )
        self.assertTrue(result["ready"])
        self.assertEqual(result["closure_ratio"], 1.0)
        self.assertEqual(result["calendar_authority_count"], 1)
        self.assertFalse(result["intersection_used_as_calendar"])

    def test_missing_and_extra_dates_are_reported_together(self):
        result = audit_asset_calendars(
            ["20260105", "20260106", "20260107"],
            {
                "l2": ["20260105", "20260107", "20260108"],
                "l4": ["20260105", "20260106"],
            },
            start="20260105",
            end="20260108",
        )
        self.assertFalse(result["ready"])
        self.assertEqual(result["assets"]["l2"]["missing_dates"], ["20260106"])
        self.assertEqual(result["assets"]["l2"]["extra_dates"], ["20260108"])
        self.assertEqual(result["assets"]["l4"]["missing_dates"], ["20260107"])

    def test_calendar_rejects_duplicates_and_invalid_dates(self):
        with self.assertRaises(CalendarContractError):
            audit_asset_calendars(
                ["20260105", "20260105"],
                {"l2": ["20260105"]},
                start="20260105",
                end="20260105",
            )
        with self.assertRaises(CalendarContractError):
            audit_asset_calendars(
                ["20260230"],
                {"l2": ["20260230"]},
                start="20260101",
                end="20261231",
            )

    def test_key_comparison_reports_all_differences(self):
        result = compare_sorted_key_rows(
            [
                ("20260105", "000001.SZ"),
                ("20260105", "000002.SZ"),
                ("20260106", "000001.SZ"),
            ],
            [
                ("20260105", "000001.SZ"),
                ("20260105", "000003.SZ"),
                ("20260106", "000001.SZ"),
            ],
        )
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["missing_key_count"], 1)
        self.assertEqual(result["extra_key_count"], 1)

    def test_key_comparison_rejects_duplicate_keys(self):
        result = compare_sorted_key_rows(
            [("20260105", "000001.SZ")],
            [
                ("20260105", "000001.SZ"),
                ("20260105", "000001.SZ"),
            ],
        )
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["duplicate_candidate_key_count"], 1)

    def test_key_comparison_rejects_unsorted_input(self):
        with self.assertRaises(CalendarContractError):
            compare_sorted_key_rows(
                [("20260106", "000001.SZ"), ("20260105", "000001.SZ")],
                [("20260105", "000001.SZ")],
            )

    def test_standalone_key_audit_rejects_duplicates(self):
        result = audit_sorted_key_rows(
            [
                ("20260105", "000001.SZ"),
                ("20260105", "000001.SZ"),
                ("20260105", "000002.SZ"),
            ]
        )
        self.assertFalse(result["unique"])
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["unique_key_count"], 2)
        self.assertEqual(result["duplicate_key_count"], 1)

    def test_t1_map_only_uses_authoritative_calendar(self):
        self.assertEqual(
            build_t1_map(["20260105", "20260106", "20260108"]),
            {
                "20260105": "20260106",
                "20260106": "20260108",
                "20260108": None,
            },
        )

    def test_descriptor_runner_checks_dates_and_key_groups(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            calendar_path = root / "calendar.json"
            calendar_path.write_text(
                json.dumps({"open_dates": ["20260105", "20260106"]}),
                encoding="utf-8",
            )
            for name in ("l2", "l4a", "l4b"):
                path = root / f"{name}.duckdb"
                connection = duckdb.connect(str(path))
                try:
                    connection.execute(
                        "CREATE TABLE data(trade_date VARCHAR, stock_code VARCHAR)"
                    )
                    connection.executemany(
                        "INSERT INTO data VALUES (?, ?)",
                        [
                            ("20260105", "000001.SZ"),
                            ("20260106", "000001.SZ"),
                        ],
                    )
                finally:
                    connection.close()
            descriptor = {
                "schema_version": 1,
                "window": {
                    "start": "20260105",
                    "end": "20260106",
                    "minimum_dates": 2,
                },
                "authoritative_calendar": {
                    "source_type": "json_dates",
                    "path": "calendar.json",
                },
                "assets": [
                    {
                        "name": "l2",
                        "source_type": "duckdb_table",
                        "path": "l2.duckdb",
                        "table": "data",
                        "date_column": "trade_date",
                    },
                    {
                        "name": "l4a",
                        "source_type": "duckdb_table",
                        "path": "l4a.duckdb",
                        "table": "data",
                        "date_column": "trade_date",
                        "key_columns": ["trade_date", "stock_code"],
                        "key_group": "l4",
                    },
                    {
                        "name": "l4b",
                        "source_type": "duckdb_table",
                        "path": "l4b.duckdb",
                        "table": "data",
                        "date_column": "trade_date",
                        "key_columns": ["trade_date", "stock_code"],
                        "key_group": "l4",
                    },
                ],
            }
            descriptor_path = root / "descriptor.json"
            descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
            result = run(descriptor_path)
        self.assertTrue(result["ready"])
        self.assertTrue(result["key_groups"]["l4"]["exact_match"])
        self.assertFalse(result["business_metrics_calculated"])

    def test_descriptor_runner_supports_nonoverlapping_recovery_parts(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "calendar.json").write_text(
                json.dumps({"open_dates": ["20260105", "20260106"]}),
                encoding="utf-8",
            )
            rows_by_file = {
                "reference.duckdb": [
                    ("20260105", "000001.SZ"),
                    ("20260106", "000001.SZ"),
                ],
                "active.duckdb": [("20260105", "000001.SZ")],
                "recovery.duckdb": [("20260106", "000001.SZ")],
            }
            for filename, rows in rows_by_file.items():
                connection = duckdb.connect(str(root / filename))
                try:
                    connection.execute(
                        "CREATE TABLE data(trade_date VARCHAR, stock_code VARCHAR)"
                    )
                    connection.executemany("INSERT INTO data VALUES (?, ?)", rows)
                finally:
                    connection.close()
            descriptor = {
                "schema_version": 1,
                "window": {
                    "start": "20260105",
                    "end": "20260106",
                    "minimum_dates": 2,
                },
                "authoritative_calendar": {
                    "source_type": "json_dates",
                    "path": "calendar.json",
                },
                "assets": [
                    {
                        "name": "reference",
                        "source_type": "duckdb_table",
                        "path": "reference.duckdb",
                        "table": "data",
                        "date_column": "trade_date",
                        "key_columns": ["trade_date", "stock_code"],
                        "key_group": "predictions",
                    },
                    {
                        "name": "active_plus_recovery",
                        "source_type": "duckdb_table",
                        "table": "data",
                        "date_column": "trade_date",
                        "key_columns": ["trade_date", "stock_code"],
                        "key_group": "predictions",
                        "parts": [
                            {"path": "active.duckdb"},
                            {"path": "recovery.duckdb"},
                        ],
                    },
                ],
            }
            descriptor_path = root / "descriptor.json"
            descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
            result = run(descriptor_path)
        self.assertTrue(result["ready"])

    def test_recovery_part_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "calendar.json").write_text(
                json.dumps({"open_dates": ["20260105"]}), encoding="utf-8"
            )
            for filename in ("reference.duckdb", "active.duckdb", "recovery.duckdb"):
                connection = duckdb.connect(str(root / filename))
                try:
                    connection.execute(
                        "CREATE TABLE data(trade_date VARCHAR, stock_code VARCHAR)"
                    )
                    connection.execute(
                        "INSERT INTO data VALUES ('20260105', '000001.SZ')"
                    )
                finally:
                    connection.close()
            descriptor = {
                "schema_version": 1,
                "window": {"start": "20260105", "end": "20260105"},
                "authoritative_calendar": {
                    "source_type": "json_dates",
                    "path": "calendar.json",
                },
                "assets": [
                    {
                        "name": "reference",
                        "source_type": "duckdb_table",
                        "path": "reference.duckdb",
                        "table": "data",
                        "date_column": "trade_date",
                        "key_columns": ["trade_date", "stock_code"],
                        "key_group": "predictions",
                    },
                    {
                        "name": "overlap",
                        "source_type": "duckdb_table",
                        "table": "data",
                        "date_column": "trade_date",
                        "key_columns": ["trade_date", "stock_code"],
                        "key_group": "predictions",
                        "parts": [
                            {"path": "active.duckdb"},
                            {"path": "recovery.duckdb"},
                        ],
                    },
                ],
            }
            descriptor_path = root / "descriptor.json"
            descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
            with self.assertRaises(CalendarContractError):
                run(descriptor_path)

    def test_standalone_asset_duplicate_key_blocks_runner(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "calendar.json").write_text(
                json.dumps({"open_dates": ["20260105"]}), encoding="utf-8"
            )
            connection = duckdb.connect(str(root / "l2.duckdb"))
            try:
                connection.execute(
                    "CREATE TABLE data(trade_date VARCHAR, stock_code VARCHAR)"
                )
                connection.executemany(
                    "INSERT INTO data VALUES (?, ?)",
                    [
                        ("20260105", "000001.SZ"),
                        ("20260105", "000001.SZ"),
                    ],
                )
            finally:
                connection.close()
            descriptor = {
                "schema_version": 1,
                "window": {"start": "20260105", "end": "20260105"},
                "authoritative_calendar": {
                    "source_type": "json_dates",
                    "path": "calendar.json",
                },
                "assets": [
                    {
                        "name": "l2",
                        "source_type": "duckdb_table",
                        "path": "l2.duckdb",
                        "table": "data",
                        "date_column": "trade_date",
                        "key_columns": ["trade_date", "stock_code"],
                    }
                ],
            }
            descriptor_path = root / "descriptor.json"
            descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
            result = run(descriptor_path)
        self.assertFalse(result["ready"])
        self.assertEqual(result["key_uniqueness"]["l2"]["duplicate_key_count"], 1)


if __name__ == "__main__":
    unittest.main()
