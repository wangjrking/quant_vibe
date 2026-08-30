from __future__ import annotations

import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from adjustment_semantics import default_adjustment_semantics, default_market_field_semantics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_rules(path: Path) -> None:
    _write_json(
        path,
        {
            "selection_rule": {
                "filter_name": "unit",
                "exclude_bj": False,
                "exclude_st_risk_warning": True,
                "exclude_delist": True,
                "skip_open_limit_up_buy": True,
                "amount_min": 1,
                "total_mv_min": 1,
                "total_mv_max": 999999999,
                "turnover_min": 0,
            },
            "model_input": {
                "weight_name": "unit",
                "entry_weights": {"3d": 0.2, "5d": 0.3, "10d": 0.5},
            },
            "position_rule": {"target_position_pct": 0.5},
            "holding_rule": {
                "holding_days": 3,
                "max_holding_days": 5,
                "score_exit_entry_ratio": 0.9,
                "min_holding_days_before_score_exit": 1,
                "score_continue_entry_ratio": 0.8,
                "holding_name": "unit",
            },
            "risk_rule": {"intraday_stop_loss_pct": 0.03, "take_profit_pct": 0.08},
        },
    )


def _strategy_manifest(strategy_dir: Path, manifests: dict[str, Path], market_db_path: str | None) -> None:
    _write_json(
        strategy_dir / "strategy_manifest.json",
        {
            "strategy_id": "prod_test",
            "input_contract": {
                "formal_manifest_3d": str(manifests["3d"]),
                "formal_manifest_5d": str(manifests["5d"]),
                "formal_manifest_10d": str(manifests["10d"]),
                "market_db_path": market_db_path,
                "allow_legacy": False,
                "adjustment_semantics": default_adjustment_semantics(),
                "market_field_semantics": default_market_field_semantics(),
            },
        },
    )


def _create_sqlite_prediction_db(path: Path) -> None:
    rows_3d = [("20260625", "000001.SZ", 0.2), ("20260625", "000002.SZ", 0.9)]
    rows_5d = [("20260625", "000001.SZ", 0.3), ("20260625", "000002.SZ", 0.8)]
    rows_10d = [("20260625", "000001.SZ", 0.4), ("20260625", "000002.SZ", 0.7)]
    with sqlite3.connect(str(path)) as conn:
        for table, rows in {
            "pred3": rows_3d,
            "pred5": rows_5d,
            "pred10": rows_10d,
        }.items():
            conn.execute(f'CREATE TABLE "{table}" (trade_date TEXT, stock_code TEXT, pred_prob REAL)')
            conn.executemany(f'INSERT INTO "{table}" VALUES (?, ?, ?)', rows)


def _create_sqlite_market_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE STOCK_DAILY_DATA (
                trade_date TEXT,
                stock_code TEXT,
                name TEXT,
                pre_close REAL,
                open REAL,
                close REAL,
                amount REAL,
                turnover_rate REAL,
                total_mv REAL,
                atr_qfq REAL,
                limit_times REAL,
                ST_TYPE REAL,
                ST_TYPE_name TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO STOCK_DAILY_DATA VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("20260625", "000001.SZ", "A", 10, 10.1, 10.2, 100000, 2.0, 1000000, 0.5, 0, 0, ""),
                ("20260625", "000002.SZ", "B", 10, 10.1, 10.2, 200000, 3.0, 2000000, 0.5, 0, 0, ""),
                ("20260626", "000001.SZ", "A", 10.2, 10.3, 10.4, 100000, 2.0, 1000000, 0.5, 0, 0, ""),
                ("20260626", "000002.SZ", "B", 10.2, 10.3, 10.4, 200000, 3.0, 2000000, 0.5, 0, 0, ""),
            ],
        )


class ExportDynamicTop1FormalSignalsTests(unittest.TestCase):
    def test_validate_strategy_output_rows_rejects_naked_market_price_names(self):
        from export_dynamic_top1_formal_signals import _validate_strategy_output_rows

        with self.assertRaisesRegex(ValueError, "naked market price fields"):
            _validate_strategy_output_rows(
                [{"stock_code": "000001.SZ", "open": 10.0, "atr_qfq": 0.5}],
                context="unit",
            )

    def test_export_signals_rejects_sqlite_sources_in_production_mode(self):
        from export_dynamic_top1_formal_signals import export_signals

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy"
            strategy_dir.mkdir(parents=True)
            pred_db = root / "predictions.db"
            market_db = root / "market.db"
            _create_sqlite_prediction_db(pred_db)
            _create_sqlite_market_db(market_db)

            manifests = {}
            for label, table in {"3d": "pred3", "5d": "pred5", "10d": "pred10"}.items():
                manifest_path = root / f"{label}.json"
                _write_json(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "sqlite_table",
                        "db_path": str(pred_db),
                        "table": table,
                        "market_db_path": str(market_db),
                        "adjustment_semantics": default_adjustment_semantics(),
                        "market_field_semantics": default_market_field_semantics(),
                    },
                )
                manifests[label] = manifest_path

            _strategy_manifest(strategy_dir, manifests, str(market_db))
            _write_rules(strategy_dir / "trading_rules.json")

            output = root / "signals.csv"
            status_output = root / "status.json"
            with self.assertRaisesRegex(ValueError, "sqlite_table"):
                export_signals(strategy_dir, output, status_output=status_output)

    def test_export_signals_supports_duckdb_sources(self):
        import duckdb

        from export_dynamic_top1_formal_signals import export_signals

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy"
            strategy_dir.mkdir(parents=True)
            duckdb_path = root / "quant_production.duckdb"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_manifest.json"
            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")

            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute('CREATE TABLE "pred3" (trade_date VARCHAR, stock_code VARCHAR, pred_prob DOUBLE)')
                conn.execute('CREATE TABLE "pred5" (trade_date VARCHAR, stock_code VARCHAR, pred_prob DOUBLE)')
                conn.execute('CREATE TABLE "pred10" (trade_date VARCHAR, stock_code VARCHAR, pred_prob DOUBLE)')
                for table, rows in {
                    "pred3": [("20260625", "000001.SZ", 0.2), ("20260625", "000002.SZ", 0.9)],
                    "pred5": [("20260625", "000001.SZ", 0.3), ("20260625", "000002.SZ", 0.8)],
                    "pred10": [("20260625", "000001.SZ", 0.4), ("20260625", "000002.SZ", 0.7)],
                }.items():
                    conn.executemany(f'INSERT INTO "{table}" VALUES (?, ?, ?)', rows)
                conn.execute(
                    """
                    CREATE TABLE "STOCK_DAILY_DATA" (
                        trade_date VARCHAR,
                        stock_code VARCHAR,
                        name VARCHAR,
                        pre_close DOUBLE,
                        open DOUBLE,
                        close DOUBLE,
                        amount DOUBLE,
                        turnover_rate DOUBLE,
                        total_mv DOUBLE,
                        atr_qfq DOUBLE,
                        limit_times DOUBLE,
                        ST_TYPE DOUBLE,
                        ST_TYPE_name VARCHAR
                    )
                    """
                )
                conn.executemany(
                    'INSERT INTO "STOCK_DAILY_DATA" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    [
                        ("20260625", "000001.SZ", "A", 10, 10.1, 10.2, 100000, 2.0, 1000000, 0.5, 0, 0, ""),
                        ("20260625", "000002.SZ", "B", 10, 10.1, 10.2, 200000, 3.0, 2000000, 0.5, 0, 0, ""),
                        ("20260626", "000001.SZ", "A", 10.2, 10.3, 10.4, 100000, 2.0, 1000000, 0.5, 0, 0, ""),
                        ("20260626", "000002.SZ", "B", 10.2, 10.3, 10.4, 200000, 3.0, 2000000, 0.5, 0, 0, ""),
                    ],
                )

            manifests = {}
            for label, table in {"3d": "pred3", "5d": "pred5", "10d": "pred10"}.items():
                manifest_path = root / f"{label}.json"
                _write_json(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(duckdb_path),
                        "table": table,
                        "market_db_path": str(duckdb_path),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": default_adjustment_semantics(),
                        "market_field_semantics": default_market_field_semantics(),
                    },
                )
                manifests[label] = manifest_path

            _strategy_manifest(strategy_dir, manifests, str(duckdb_path))
            _write_rules(strategy_dir / "trading_rules.json")

            output = root / "signals.csv"
            status_output = root / "status.json"
            result = export_signals(strategy_dir, output, status_output=status_output)

            self.assertEqual(result["rows"][0]["stock_code"], "000002.SZ")
            self.assertEqual(result["status"]["buy_date"], "20260626")
            self.assertTrue(status_output.is_file())

    def test_export_signals_requires_market_field_semantics_in_strategy_contract(self):
        from export_dynamic_top1_formal_signals import export_signals

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy"
            strategy_dir.mkdir(parents=True)
            duckdb_path = root / "quant_production.duckdb"
            audit_record = root / "audit.md"
            migration_manifest = root / "duckdb_manifest.json"
            audit_record.write_text("approved", encoding="utf-8")
            migration_manifest.write_text("{}", encoding="utf-8")

            import duckdb

            with duckdb.connect(str(duckdb_path)) as conn:
                conn.execute('CREATE TABLE "pred3" (trade_date VARCHAR, stock_code VARCHAR, pred_prob DOUBLE)')
                conn.execute('CREATE TABLE "pred5" (trade_date VARCHAR, stock_code VARCHAR, pred_prob DOUBLE)')
                conn.execute('CREATE TABLE "pred10" (trade_date VARCHAR, stock_code VARCHAR, pred_prob DOUBLE)')
                conn.executemany(
                    'INSERT INTO "pred3" VALUES (?, ?, ?)',
                    [("20260625", "000001.SZ", 0.2)],
                )
                conn.executemany(
                    'INSERT INTO "pred5" VALUES (?, ?, ?)',
                    [("20260625", "000001.SZ", 0.3)],
                )
                conn.executemany(
                    'INSERT INTO "pred10" VALUES (?, ?, ?)',
                    [("20260625", "000001.SZ", 0.4)],
                )
                conn.execute(
                    """
                    CREATE TABLE "STOCK_DAILY_DATA" (
                        trade_date VARCHAR,
                        stock_code VARCHAR,
                        name VARCHAR,
                        pre_close DOUBLE,
                        open DOUBLE,
                        close DOUBLE,
                        amount DOUBLE,
                        turnover_rate DOUBLE,
                        total_mv DOUBLE,
                        atr_qfq DOUBLE,
                        limit_times DOUBLE,
                        ST_TYPE DOUBLE,
                        ST_TYPE_name VARCHAR
                    )
                    """
                )
                conn.executemany(
                    'INSERT INTO "STOCK_DAILY_DATA" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    [
                        ("20260625", "000001.SZ", "A", 10, 10.1, 10.2, 100000, 2.0, 1000000, 0.5, 0, 0, ""),
                        ("20260626", "000001.SZ", "A", 10.2, 10.3, 10.4, 100000, 2.0, 1000000, 0.5, 0, 0, ""),
                    ],
                )

            manifests = {}
            for label, table in {"3d": "pred3", "5d": "pred5", "10d": "pred10"}.items():
                manifest_path = root / f"{label}.json"
                _write_json(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "duckdb_table",
                        "db_path": str(duckdb_path),
                        "table": table,
                        "market_db_path": str(duckdb_path),
                        "audit_record": str(audit_record),
                        "duckdb_migration_manifest": str(migration_manifest),
                        "adjustment_semantics": default_adjustment_semantics(),
                        "market_field_semantics": default_market_field_semantics(),
                    },
                )
                manifests[label] = manifest_path

            _write_json(
                strategy_dir / "strategy_manifest.json",
                {
                    "strategy_id": "prod_test",
                    "input_contract": {
                        "formal_manifest_3d": str(manifests["3d"]),
                        "formal_manifest_5d": str(manifests["5d"]),
                        "formal_manifest_10d": str(manifests["10d"]),
                        "market_db_path": str(duckdb_path),
                        "allow_legacy": False,
                        "adjustment_semantics": default_adjustment_semantics(),
                    },
                },
            )
            _write_rules(strategy_dir / "trading_rules.json")

            with self.assertRaisesRegex(ValueError, "missing market_field_semantics"):
                export_signals(strategy_dir, root / "signals.csv")


if __name__ == "__main__":
    unittest.main()
