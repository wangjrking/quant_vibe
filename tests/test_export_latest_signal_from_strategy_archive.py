from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from export_latest_signal_from_strategy_archive import export_latest_signal


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_signal_csv(path: Path, *, buy_day_hard_gate_complete: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "signal_date",
                "buy_date",
                "stock_code",
                "symbol",
                "rank",
                "buy_day_hard_gate_complete",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "signal_date": "20260630",
                "buy_date": "20260701",
                "stock_code": "000001.SZ",
                "symbol": "SZSE.000001",
                "rank": "1",
                "buy_day_hard_gate_complete": buy_day_hard_gate_complete,
            }
        )


def _write_ambiguous_signal_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "signal_date",
                "buy_date",
                "stock_code",
                "symbol",
                "rank",
                "atr",
                "buy_day_hard_gate_complete",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "signal_date": "20260630",
                "buy_date": "20260701",
                "stock_code": "000001.SZ",
                "symbol": "SZSE.000001",
                "rank": "1",
                "atr": "0.2",
                "buy_day_hard_gate_complete": "False",
            }
        )


def _write_ambiguous_raw_return_signal_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "signal_date",
                "buy_date",
                "stock_code",
                "symbol",
                "rank",
                "pct_chg",
                "buy_day_hard_gate_complete",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "signal_date": "20260630",
                "buy_date": "20260701",
                "stock_code": "000001.SZ",
                "symbol": "SZSE.000001",
                "rank": "1",
                "pct_chg": "2.0",
                "buy_day_hard_gate_complete": "False",
            }
        )


class ExportLatestSignalFromStrategyArchiveTests(unittest.TestCase):
    def test_status_is_pending_when_buy_day_hard_gate_not_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy"
            archive_csv = strategy_dir / "signals" / "full_history.csv"
            output_csv = root / "latest.csv"
            status_json = root / "latest_status.json"
            _write_signal_csv(archive_csv, buy_day_hard_gate_complete="False")
            _write_json(
                strategy_dir / "strategy_manifest.json",
                {
                    "strategy_id": "prod_test",
                    "full_history_signal_file": str(archive_csv),
                    "current_signal": {
                        "buy_day_hard_gate_complete": False,
                    },
                },
            )

            result = export_latest_signal(
                strategy_dir=strategy_dir,
                output=output_csv,
                status_output=status_json,
                require_fresh=False,
            )

            self.assertEqual(result["status"]["status"], "pending_buy_day_hard_gate")

    def test_status_is_ready_when_buy_day_hard_gate_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy"
            archive_csv = strategy_dir / "signals" / "full_history.csv"
            output_csv = root / "latest.csv"
            status_json = root / "latest_status.json"
            _write_signal_csv(archive_csv, buy_day_hard_gate_complete="True")
            _write_json(
                strategy_dir / "strategy_manifest.json",
                {
                    "strategy_id": "prod_test",
                    "full_history_signal_file": str(archive_csv),
                    "current_signal": {
                        "buy_day_hard_gate_complete": True,
                    },
                },
            )

            result = export_latest_signal(
                strategy_dir=strategy_dir,
                output=output_csv,
                status_output=status_json,
                require_fresh=False,
            )

            self.assertEqual(result["status"]["status"], "ready_for_human_confirmation_execution")

    def test_export_rejects_ambiguous_front_adjusted_field_names_in_archive_signal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy"
            archive_csv = strategy_dir / "signals" / "full_history.csv"
            output_csv = root / "latest.csv"
            _write_ambiguous_signal_csv(archive_csv)
            _write_json(
                strategy_dir / "strategy_manifest.json",
                {
                    "strategy_id": "prod_test",
                    "full_history_signal_file": str(archive_csv),
                    "current_signal": {
                        "buy_day_hard_gate_complete": False,
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "naked front-adjusted indicator"):
                export_latest_signal(
                    strategy_dir=strategy_dir,
                    output=output_csv,
                    require_fresh=False,
                )

    def test_export_rejects_ambiguous_raw_return_field_names_in_archive_signal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy_dir = root / "strategy"
            archive_csv = strategy_dir / "signals" / "full_history.csv"
            output_csv = root / "latest.csv"
            _write_ambiguous_raw_return_signal_csv(archive_csv)
            _write_json(
                strategy_dir / "strategy_manifest.json",
                {
                    "strategy_id": "prod_test",
                    "full_history_signal_file": str(archive_csv),
                    "current_signal": {
                        "buy_day_hard_gate_complete": False,
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "naked raw-derived market fields"):
                export_latest_signal(
                    strategy_dir=strategy_dir,
                    output=output_csv,
                    require_fresh=False,
                )


if __name__ == "__main__":
    unittest.main()
