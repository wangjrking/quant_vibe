import csv
import json
import os
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))


def _load_module(testcase: unittest.TestCase):
    try:
        return import_module("build_manual_trade_package")
    except ModuleNotFoundError as exc:  # pragma: no cover - red phase guard
        testcase.fail(f"build_manual_trade_package module missing: {exc}")


class BuildManualTradePackageTests(unittest.TestCase):
    def test_load_latest_signal_batch_only_returns_latest_signal_date(self):
        module = _load_module(self)
        with tempfile.TemporaryDirectory() as tmpdir:
            signal_path = Path(tmpdir) / "prod_a_latest.csv"
            with signal_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["signal_date", "buy_date", "stock_code", "name", "rank", "target_pct"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "signal_date": "20260617",
                        "buy_date": "20260618",
                        "stock_code": "000001.SZ",
                        "name": "旧信号",
                        "rank": "1",
                        "target_pct": "0.50",
                    }
                )
                writer.writerow(
                    {
                        "signal_date": "20260618",
                        "buy_date": "20260622",
                        "stock_code": "000002.SZ",
                        "name": "新信号A",
                        "rank": "1",
                        "target_pct": "0.30",
                    }
                )
                writer.writerow(
                    {
                        "signal_date": "20260618",
                        "buy_date": "20260622",
                        "stock_code": "000003.SZ",
                        "name": "新信号B",
                        "rank": "2",
                        "target_pct": "0.20",
                    }
                )

            rows = module.load_latest_signal_batch(signal_path)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["stock_code"] for row in rows}, {"000002.SZ", "000003.SZ"})
        self.assertTrue(all(row["signal_date"] == "20260618" for row in rows))

    def test_build_trade_package_classifies_buy_sell_and_hold(self):
        module = _load_module(self)
        registry = {
            "production": {
                "current": "prod_a",
                "strategies": [
                    {
                        "strategy_id": "prod_a",
                        "name": "测试投产策略",
                        "status": "production",
                        "version": "v1",
                        "path": "strategy_library/production/prod_a",
                    }
                ]
            }
        }
        strategy_manifest = {
            "strategy_id": "prod_a",
            "name": "测试投产策略",
            "production_version": "V1.0",
            "version": "v1",
        }
        signal_rows = [
            {
                "signal_date": "20260618",
                "buy_date": "20260622",
                "stock_code": "000001.SZ",
                "name": "继续持有股",
                "rank": "1",
                "target_pct": "0.21",
            },
            {
                "signal_date": "20260618",
                "buy_date": "20260622",
                "stock_code": "000002.SZ",
                "name": "新买入股A",
                "rank": "2",
                "target_pct": "0.20",
            },
            {
                "signal_date": "20260618",
                "buy_date": "20260622",
                "stock_code": "000003.SZ",
                "name": "新买入股B",
                "rank": "3",
                "target_pct": "0.19",
            },
        ]
        holdings_summary = {
            "ok": True,
            "account_id": "8883530863",
            "account_type": "STOCK",
            "asset": {
                "total_asset": 100000.0,
                "market_value": 32000.0,
                "cash": 68000.0,
                "fetch_balance": 68000.0,
            },
            "positions": [
                {
                    "stock_code": "000001.SZ",
                    "instrument_name": "继续持有股",
                    "volume": 1000,
                    "can_use_volume": 1000,
                    "open_price": 21.0,
                    "last_price": 21.2,
                    "market_value": 21200.0,
                    "position_profit": 200.0,
                    "profit_rate": 0.01,
                },
                {
                    "stock_code": "000009.SZ",
                    "instrument_name": "卖出股",
                    "volume": 2000,
                    "can_use_volume": 2000,
                    "open_price": 5.0,
                    "last_price": 5.4,
                    "market_value": 10800.0,
                    "position_profit": 800.0,
                    "profit_rate": 0.08,
                },
            ],
        }

        package = module.build_trade_package(
            registry=registry,
            strategy_manifest=strategy_manifest,
            signal_rows=signal_rows,
            holdings_summary=holdings_summary,
            platform="QMT",
            account_label="模拟盘",
        )

        self.assertEqual(package["strategy"]["strategy_id"], "prod_a")
        self.assertEqual(package["signal"]["signal_date"], "20260618")
        self.assertEqual(package["signal"]["buy_date"], "20260622")
        self.assertEqual(len(package["actions"]["buy"]), 2)
        self.assertEqual(len(package["actions"]["sell"]), 1)
        self.assertEqual(len(package["actions"]["hold"]), 1)
        self.assertEqual(package["actions"]["hold"][0]["stock_code"], "000001.SZ")
        self.assertEqual(package["actions"]["sell"][0]["stock_code"], "000009.SZ")
        self.assertEqual(
            {row["stock_code"] for row in package["actions"]["buy"]},
            {"000002.SZ", "000003.SZ"},
        )

    def test_build_trade_package_falls_back_to_code_name_when_display_name_is_garbled(self):
        module = _load_module(self)
        registry = {
            "production": {
                "current": "prod_a",
                "strategies": [
                    {
                        "strategy_id": "prod_a",
                        "name": "????",
                        "status": "production",
                        "version": "v1",
                        "path": "strategy_library/production/prod_a",
                    }
                ]
            }
        }
        strategy_manifest = {
            "strategy_id": "prod_a",
            "name": "??5D10D pred_gap ??????",
            "code_name": "formal_5d10d_gap_enhanced",
            "production_version": "V1.0",
            "version": "v1",
        }
        signal_rows = [
            {
                "signal_date": "20260618",
                "buy_date": "20260622",
                "stock_code": "000001.SZ",
                "name": "继续持有股",
                "rank": "1",
                "target_pct": "0.21",
            }
        ]
        holdings_summary = {
            "ok": True,
            "account_id": "8883530863",
            "account_type": "STOCK",
            "asset": {
                "total_asset": 100000.0,
                "market_value": 21000.0,
                "cash": 79000.0,
                "fetch_balance": 79000.0,
            },
            "positions": [
                {
                    "stock_code": "000001.SZ",
                    "instrument_name": "继续持有股",
                    "volume": 1000,
                    "can_use_volume": 1000,
                    "open_price": 21.0,
                    "last_price": 21.0,
                    "market_value": 21000.0,
                    "position_profit": 0.0,
                    "profit_rate": 0.0,
                }
            ],
        }

        package = module.build_trade_package(
            registry=registry,
            strategy_manifest=strategy_manifest,
            signal_rows=signal_rows,
            holdings_summary=holdings_summary,
            platform="QMT",
            account_label="模拟盘",
        )

        self.assertEqual(package["strategy"]["name"], "formal_5d10d_gap_enhanced")

    def test_write_trade_package_outputs_markdown_csv_and_json(self):
        module = _load_module(self)
        package = {
            "strategy": {
                "strategy_id": "prod_a",
                "name": "测试投产策略",
                "production_version": "V1.0",
                "version": "v1",
            },
            "signal": {
                "signal_date": "20260618",
                "buy_date": "20260622",
                "source_file": "prod_a_latest.csv",
            },
            "account": {
                "platform": "QMT",
                "account_label": "模拟盘",
                "account_id": "8883530863",
                "total_asset": 100000.0,
                "cash": 68000.0,
                "market_value": 32000.0,
            },
            "summary": {
                "buy_count": 1,
                "sell_count": 1,
                "hold_count": 1,
            },
            "actions": {
                "buy": [
                    {
                        "stock_code": "000002.SZ",
                        "stock_name": "新买入股A",
                        "rank": 1,
                        "target_pct": 0.2,
                        "planned_amount": 20000.0,
                    }
                ],
                "sell": [
                    {
                        "stock_code": "000009.SZ",
                        "stock_name": "卖出股",
                        "current_shares": 2000,
                        "available_shares": 2000,
                        "sell_reason": "不在最新目标池",
                    }
                ],
                "hold": [
                    {
                        "stock_code": "000001.SZ",
                        "stock_name": "继续持有股",
                        "current_position_pct": 0.212,
                        "hold_reason": "已在最新目标池，且仓位偏差在容忍范围内",
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = module.write_trade_package(package, Path(tmpdir))

            ticket_text = Path(outputs["ticket_path"]).read_text(encoding="utf-8")
            csv_text = Path(outputs["csv_path"]).read_text(encoding="utf-8-sig")
            summary = json.loads(Path(outputs["json_path"]).read_text(encoding="utf-8"))

        self.assertIn("次日交易执行单", ticket_text)
        self.assertIn("测试投产策略", ticket_text)
        self.assertIn("000002.SZ", csv_text)
        self.assertEqual(summary["strategy"]["strategy_id"], "prod_a")


    def test_build_from_runtime_supports_duckdb_strategy_assets(self):
        import duckdb
        import pandas as pd

        module = _load_module(self)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            duckdb_path = root / "quant_production.duckdb"
            holdings_path = root / "holdings.json"
            holdings_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "account_id": "acct",
                        "account_type": "STOCK",
                        "asset": {"total_asset": 100000.0, "market_value": 0.0, "cash": 100000.0},
                        "positions": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with duckdb.connect(str(duckdb_path)) as conn:
                conn.register(
                    "_registry_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "is_current_production": True,
                                "registry_payload_json": json.dumps(
                                    {
                                        "strategy_id": "prod_a",
                                        "name": "测试策略",
                                        "status": "production",
                                        "path": "strategy_library/production/prod_a",
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l5_strategy_registry_current" AS SELECT * FROM _registry_df')
                conn.unregister("_registry_df")
                conn.register(
                    "_manifest_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "manifest_payload_json": json.dumps(
                                    {
                                        "strategy_id": "prod_a",
                                        "name": "测试策略",
                                        "production_version": "v1",
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l5_strategy_manifest_current" AS SELECT * FROM _manifest_df')
                conn.unregister("_manifest_df")
                conn.register(
                    "_signal_rows_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "file_name": "prod_a_latest.csv",
                                "row_index": 1,
                                "signal_date": "20260628",
                                "rank": "1",
                                "row_payload_json": json.dumps(
                                    {
                                        "signal_date": "20260628",
                                        "buy_date": "20260630",
                                        "stock_code": "000001.SZ",
                                        "name": "A",
                                        "rank": "1",
                                        "target_pct": "1.0",
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l7_signal_rows_current" AS SELECT * FROM _signal_rows_df')
                conn.unregister("_signal_rows_df")
                conn.register(
                    "_signal_files_df",
                    pd.DataFrame(
                        [
                            {
                                "strategy_id": "prod_a",
                                "file_name": "prod_a_latest.csv",
                                "source_path": "quant/data_file/production_signals/prod_a_latest.csv",
                                "modified_at": "2026-06-28T10:00:00",
                            }
                        ]
                    ),
                )
                conn.execute('CREATE TABLE "prod_l7_signal_files_current" AS SELECT * FROM _signal_files_df')
                conn.unregister("_signal_files_df")

            old_env = dict(os.environ)
            os.environ["QUANT_STRATEGY_ASSET_BACKEND"] = "duckdb"
            os.environ["QUANT_STRATEGY_DUCKDB"] = str(duckdb_path)
            try:
                package = module.build_from_runtime(
                    holdings_json=holdings_path,
                    platform="QMT",
                    account_label="模拟盘",
                )
            finally:
                os.environ.clear()
                os.environ.update(old_env)

        self.assertEqual(package["strategy"]["strategy_id"], "prod_a")
        self.assertEqual(package["signal"]["signal_date"], "20260628")
        self.assertEqual(package["signal"]["source_file"], "quant/data_file/production_signals/prod_a_latest.csv")


if __name__ == "__main__":
    unittest.main()
