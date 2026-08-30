from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))


from build_l7_delivery_package import build_l7_delivery_package


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class BuildL7DeliveryPackageTests(unittest.TestCase):
    def test_build_l7_delivery_package_uses_action_latest_when_archive_path_is_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "prod_test"
            production_signal_dir = root / "data_file" / "production_signals"
            delivery_root = root / "data_file" / "runtime" / "trading_agent" / "delivery_packages"
            market_db_path = root / "market.duckdb"
            latest_csv = production_signal_dir / "prod_test_latest.csv"
            latest_status = production_signal_dir / "prod_test_latest_status.json"

            _write_json(root / "strategy_library" / "registry.json", {"production": {"current": "prod_test", "strategies": [{"strategy_id": "prod_test", "status": "production"}]}})
            _write_json(strategy_dir / "strategy_manifest.json", {"strategy_id": "prod_test", "current_signal": {"signal_date": "20260731", "buy_date": "20260803"}})
            rows = [
                {"strategy_id": "prod_test", "signal_date": "20260731", "buy_date": "20260803", "action": "BUY", "stock_code": "000001.SZ", "name": "测试买入", "market": "深市", "strategy_score": "0.99", "score_rank": "1", "score_denominator": "2", "target_pct": "0.2", "reason": "test", "execution_open_raw": "", "status": "pending_buy_day_hard_gate"},
                {"strategy_id": "prod_test", "signal_date": "20260731", "buy_date": "20260803", "action": "SELL", "stock_code": "600000.SH", "name": "测试卖出", "market": "沪市", "strategy_score": "0.85", "score_rank": "2", "score_denominator": "2", "target_pct": "0", "reason": "test", "execution_open_raw": "", "status": "pending_buy_day_hard_gate"},
            ]
            _write_csv(latest_csv, rows)
            _write_json(latest_status, {"strategy_id": "prod_test", "status": "pending_buy_day_hard_gate", "signal_semantics": "trade_actions_pending_buy_day_hard_gate", "signal_date": "20260731", "buy_date": "20260803", "no_signal": False, "hold_only": False, "action_count": 2, "row_count": 2, "buy_count": 1, "sell_count": 1, "l7_execution_allowed": False, "execution_allowed": False, "approved_for_execution": False, "auto_execution_allowed": False, "live_execution_allowed": False, "formal_batch_generated": True})
            with duckdb.connect(str(market_db_path)) as conn:
                conn.execute("CREATE TABLE STOCK_DAILY_DATA (trade_date VARCHAR, stock_code VARCHAR)")

            result = build_l7_delivery_package(strategy_dir=strategy_dir, production_signal_dir=production_signal_dir, delivery_root=delivery_root, market_db_path=market_db_path, signal_date="20260731", buy_date="20260803", sync_duckdb=False)
            summary = json.loads((Path(result["delivery_dir"]) / "l7_delivery_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(summary["signal_semantics"], "action_signals")
            self.assertFalse(summary["no_signal"])
            self.assertFalse(summary["l7_execution_allowed"])
            self.assertFalse(summary["ready_for_human_confirmation_execution"])

    def test_build_l7_delivery_package_creates_latest_status_and_pending_hard_gate_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "prod_test"
            production_signal_dir = root / "data_file" / "production_signals"
            delivery_root = root / "data_file" / "runtime" / "trading_agent" / "delivery_packages"
            market_db_path = root / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

            archive_latest_csv = strategy_dir / "signals" / "latest_signal_20260703_for_20260706.csv"
            archive_full_history_csv = strategy_dir / "signals" / "full_history_prod_test.csv"

            _write_json(
                root / "strategy_library" / "registry.json",
                {
                    "production": {
                        "current": "prod_test",
                        "strategies": [{"strategy_id": "prod_test", "status": "production"}],
                    }
                },
            )

            rows = [
                {
                    "signal_date": "20260703",
                    "buy_date": "20260706",
                    "symbol": "SZSE.000001",
                    "stock_code": "000001.SZ",
                    "name": "平安银行",
                    "rank": "1",
                    "pred_prob": "0.7",
                    "entry_score": "0.7",
                    "target_pct": "0.5",
                    "buy_day_market_available": "False",
                    "buy_day_hard_gate_complete": "False",
                    "buy_day_st_rejected": "False",
                    "buy_day_open_limit_up_rejected": "False",
                    "latest_market_date": "",
                },
                {
                    "signal_date": "20260703",
                    "buy_date": "20260706",
                    "symbol": "SHSE.600000",
                    "stock_code": "600000.SH",
                    "name": "浦发银行",
                    "rank": "2",
                    "pred_prob": "0.6",
                    "entry_score": "0.6",
                    "target_pct": "0.4",
                    "buy_day_market_available": "False",
                    "buy_day_hard_gate_complete": "False",
                    "buy_day_st_rejected": "False",
                    "buy_day_open_limit_up_rejected": "False",
                    "latest_market_date": "",
                },
            ]
            _write_csv(archive_latest_csv, rows)
            _write_csv(archive_full_history_csv, rows)
            _write_json(
                strategy_dir / "strategy_manifest.json",
                {
                    "strategy_id": "prod_test",
                    "status": "production",
                    "full_history_signal_file": str(archive_full_history_csv),
                    "latest_signal_file": str(archive_latest_csv),
                    "latest_signal_status": "pending_buy_day_hard_gate",
                    "current_signal": {
                        "signal_date": "20260703",
                        "buy_date": "20260706",
                        "buy_day_market_available": False,
                        "buy_day_hard_gate_complete": False,
                        "l7_execution_allowed": False,
                        "status": "pending_buy_day_hard_gate",
                    },
                    "input_contract": {
                        "formal_manifest_3d": "",
                        "formal_manifest_5d": "",
                        "formal_manifest_10d": "",
                    },
                },
            )

            market_db_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(market_db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE STOCK_DAILY_DATA (
                        trade_date VARCHAR,
                        stock_code VARCHAR,
                        name VARCHAR,
                        open DOUBLE,
                        pre_close DOUBLE,
                        ST_TYPE INTEGER,
                        ST_TYPE_name VARCHAR,
                        limit_times INTEGER
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO STOCK_DAILY_DATA VALUES
                    ('20260703', '000001.SZ', '平安银行', 10.0, 9.8, 0, '', 0),
                    ('20260703', '600000.SH', '浦发银行', 9.5, 9.4, 0, '', 0)
                    """
                )

            result = build_l7_delivery_package(
                strategy_dir=strategy_dir,
                production_signal_dir=production_signal_dir,
                delivery_root=delivery_root,
                market_db_path=market_db_path,
            )

            latest_status_path = production_signal_dir / "prod_test_latest_status.json"
            self.assertTrue(latest_status_path.exists())
            latest_status = json.loads(latest_status_path.read_text(encoding="utf-8"))
            self.assertEqual(latest_status["status"], "pending_buy_day_hard_gate")
            self.assertEqual(latest_status["signal_date"], "20260703")
            self.assertEqual(latest_status["buy_date"], "20260706")

            delivery_dir = delivery_root / "prod_test_20260703_for_20260706"
            self.assertEqual(Path(result["delivery_dir"]), delivery_dir)
            summary_path = delivery_dir / "l7_delivery_summary.json"
            hard_gate_summary_path = delivery_dir / "buy_day_hard_gate_current_check" / "buy_day_hard_gate_summary.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(hard_gate_summary_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            hard_gate_summary = json.loads(hard_gate_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["strategy_id"], "prod_test")
            self.assertEqual(summary["signal_date"], "20260703")
            self.assertEqual(summary["buy_date"], "20260706")
            self.assertEqual(summary["latest_status"]["status"], "pending_buy_day_hard_gate")
            self.assertEqual(hard_gate_summary["status"], "pending_buy_day_hard_gate")
            self.assertEqual(str(hard_gate_summary["market_db_path"]).lower(), str(market_db_path).lower())
            self.assertEqual(hard_gate_summary["signal_count"], 2)

    def test_build_l7_delivery_package_accepts_strict_header_only_no_signal_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "prod_test"
            production_signal_dir = root / "data_file" / "production_signals"
            delivery_root = root / "data_file" / "runtime" / "trading_agent" / "delivery_packages"
            market_db_path = root / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            latest_csv = production_signal_dir / "prod_test_latest.csv"
            latest_status = production_signal_dir / "prod_test_latest_status.json"

            _write_json(
                root / "strategy_library" / "registry.json",
                {
                    "production": {
                        "current": "prod_test",
                        "strategies": [{"strategy_id": "prod_test", "status": "production"}],
                    }
                },
            )
            _write_json(
                strategy_dir / "strategy_manifest.json",
                {
                    "strategy_id": "prod_test",
                    "status": "production",
                    "current_signal": {
                        "signal_date": "20260729",
                        "buy_date": "20260730",
                    },
                },
            )
            fieldnames = [
                "strategy_id",
                "signal_date",
                "buy_date",
                "action",
                "stock_code",
                "name",
                "market",
                "strategy_score",
                "score_rank",
                "score_denominator",
                "target_pct",
                "reason",
                "execution_open_raw",
                "status",
            ]
            latest_csv.parent.mkdir(parents=True, exist_ok=True)
            with latest_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
            _write_json(
                latest_status,
                {
                    "strategy_id": "prod_test",
                    "status": "pending_buy_day_hard_gate",
                    "signal_semantics": "no_signal_hold_only",
                    "signal_date": "20260729",
                    "buy_date": "20260730",
                    "no_signal": True,
                    "hold_only": True,
                    "action_count": 0,
                    "row_count": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "l7_execution_allowed": False,
                    "execution_allowed": False,
                    "approved_for_execution": False,
                    "auto_execution_allowed": False,
                    "live_execution_allowed": False,
                    "approval_status": "pending_l5_l6_audit",
                    "formal_batch_generated": True,
                },
            )
            market_db_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(market_db_path)) as conn:
                conn.execute("CREATE TABLE STOCK_DAILY_DATA (trade_date VARCHAR, stock_code VARCHAR)")

            result = build_l7_delivery_package(
                strategy_dir=strategy_dir,
                production_signal_dir=production_signal_dir,
                delivery_root=delivery_root,
                market_db_path=market_db_path,
                signal_date="20260729",
                buy_date="20260730",
                sync_duckdb=False,
            )

            delivery_dir = Path(result["delivery_dir"])
            summary = json.loads((delivery_dir / "l7_delivery_summary.json").read_text(encoding="utf-8"))
            hard_gate = json.loads(
                (delivery_dir / "buy_day_hard_gate_current_check" / "buy_day_hard_gate_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            with (delivery_dir / "l7_platform_signals.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(list(reader), [])
                self.assertIn("action", reader.fieldnames or [])

            self.assertEqual(summary["row_count"], 0)
            self.assertTrue(summary["no_signal"])
            self.assertTrue(summary["hold_only"])
            self.assertEqual(summary["signal_semantics"], "no_signal_hold_only")
            self.assertEqual(summary["status"], "pending_buy_day_hard_gate")
            self.assertFalse(summary["ready_for_human_confirmation_execution"])
            self.assertFalse(summary["pending_user_approval"])
            self.assertFalse(summary["auto_executable"])
            self.assertFalse(summary["live_trading_ready"])
            self.assertFalse(summary["l7_execution_allowed"])
            self.assertTrue(summary["integrity_checks"]["row_count_valid"])
            self.assertEqual(hard_gate["status"], "pending_buy_day_hard_gate")
            self.assertEqual(hard_gate["signal_semantics"], "no_signal_hold_only")
            self.assertFalse(hard_gate["buy_day_hard_gate_applicable"])
            self.assertFalse(hard_gate["ready_for_human_confirmation_execution"])
            self.assertFalse(hard_gate["execution_allowed"])

            with self.assertRaisesRegex(ValueError, "requires explicit upstream audit approval"):
                build_l7_delivery_package(
                    strategy_dir=strategy_dir,
                    production_signal_dir=production_signal_dir,
                    delivery_root=delivery_root,
                    market_db_path=market_db_path,
                    signal_date="20260729",
                    buy_date="20260730",
                    sync_duckdb=True,
                )
            self.assertFalse((root / "data_file" / "production_assets" / "duckdb" / "production" / "l7").exists())

            approved_result = build_l7_delivery_package(
                strategy_dir=strategy_dir,
                production_signal_dir=production_signal_dir,
                delivery_root=delivery_root,
                market_db_path=market_db_path,
                signal_date="20260729",
                buy_date="20260730",
                sync_duckdb=True,
                no_signal_upstream_audit_approved=True,
            )
            sync_result = approved_result["sync_result"]
            self.assertEqual(sync_result["signal_row_count"], 0)
            self.assertEqual(sync_result["status_row_count"], 1)
            with duckdb.connect(str(sync_result["signal_status_duckdb_path"]), read_only=True) as conn:
                payload = json.loads(
                    conn.execute('SELECT payload_json FROM "prod_l7_signal_status_current"').fetchone()[0]
                )
            self.assertFalse(payload["ready_for_human_confirmation_execution"])
            self.assertFalse(payload["pending_user_approval"])
            self.assertFalse(payload["auto_executable"])
            self.assertFalse(payload["live_trading_ready"])

    def test_build_l7_delivery_package_rejects_unstructured_empty_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "prod_test"
            production_signal_dir = root / "data_file" / "production_signals"
            latest_csv = production_signal_dir / "prod_test_latest.csv"
            latest_status = production_signal_dir / "prod_test_latest_status.json"

            _write_json(
                root / "strategy_library" / "registry.json",
                {
                    "production": {
                        "current": "prod_test",
                        "strategies": [{"strategy_id": "prod_test", "status": "production"}],
                    }
                },
            )
            _write_json(strategy_dir / "strategy_manifest.json", {"strategy_id": "prod_test"})
            latest_csv.parent.mkdir(parents=True, exist_ok=True)
            latest_csv.write_text("signal_date,buy_date,action\n", encoding="utf-8-sig")
            _write_json(
                latest_status,
                {
                    "strategy_id": "prod_test",
                    "status": "pending_buy_day_hard_gate",
                    "signal_date": "20260729",
                    "buy_date": "20260730",
                    "row_count": 0,
                },
            )

            with self.assertRaisesRegex(ValueError, "signal_semantics=no_signal_hold_only"):
                build_l7_delivery_package(
                    strategy_dir=strategy_dir,
                    production_signal_dir=production_signal_dir,
                    delivery_root=root / "delivery",
                    market_db_path=root / "market.duckdb",
                    signal_date="20260729",
                    buy_date="20260730",
                    sync_duckdb=False,
                )

    def test_build_l7_delivery_package_fails_closed_without_production_current(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy_library" / "production" / "withdrawn_strategy"
            production_signal_dir = root / "data_file" / "production_signals"
            delivery_root = root / "data_file" / "runtime" / "trading_agent" / "delivery_packages"
            market_db_path = root / "data_file" / "market.duckdb"
            _write_json(
                root / "strategy_library" / "registry.json",
                {
                    "production": {
                        "current": "",
                        "state": "no_available_production_strategy",
                        "strategies": [],
                    }
                },
            )
            _write_json(
                strategy_dir / "strategy_manifest.json",
                {"strategy_id": "withdrawn_strategy", "status": "withdrawn_unreproducible"},
            )

            with self.assertRaisesRegex(ValueError, "production.current missing"):
                build_l7_delivery_package(
                    strategy_dir=strategy_dir,
                    production_signal_dir=production_signal_dir,
                    delivery_root=delivery_root,
                    market_db_path=market_db_path,
                )

            self.assertFalse(production_signal_dir.exists())
            self.assertFalse(delivery_root.exists())


if __name__ == "__main__":
    unittest.main()
