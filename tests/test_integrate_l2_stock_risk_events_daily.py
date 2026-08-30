from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

import integrate_l2_stock_risk_events_daily as daily
from tushare_stock_risk_event_contract import API_SPEC_BY_NAME, normalize_source_frame


class FakePro:
    def __init__(self, dates: list[str]) -> None:
        self.dates = dates

    def query(self, api: str, **kwargs):
        if api != "trade_cal":
            raise AssertionError(api)
        return pd.DataFrame({"cal_date": self.dates, "is_open": [1] * len(self.dates)})


def write_one_table(path: Path, table: str, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    try:
        connection.register("payload", frame)
        connection.execute(f'CREATE TABLE "{table}" AS SELECT * FROM payload')
    finally:
        connection.close()


class L2StockRiskEventDailyTests(unittest.TestCase):
    def test_target_date_update_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            l1_root = data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
            output_root = root / "out"
            bootstrap = root / "bootstrap"
            date = "20260821"
            shock = normalize_source_frame(
                "stk_shock",
                pd.DataFrame(
                    [["000001.SZ", date, "A", "深交所", "reason", "period"]],
                    columns=API_SPEC_BY_NAME["stk_shock"].required_columns,
                ),
            )
            for api in ("stk_shock", "stk_high_shock", "stk_alert"):
                frame = shock if api == "stk_shock" else normalize_source_frame(api, pd.DataFrame())
                write_one_table(l1_root / f"{api}.duckdb", api, frame)

            source_root = (
                Path(r"D:/work/quant/quant_mcp/quant/data_file/experimental_assets")
                / "tushare_stock_risk_events_v1"
            )
            bootstrap.mkdir(parents=True)
            shutil.copy2(source_root / "l2_stock_risk_events.duckdb", bootstrap)
            shutil.copy2(source_root / "l2_stock_risk_signal.duckdb", bootstrap)
            dates = pd.bdate_range("2026-02-01", "2026-09-30").strftime("%Y%m%d").tolist()

            first = daily.integrate(
                target_trade_date=date,
                data_dir=data_dir,
                output_root=output_root,
                pro=FakePro(dates),
                bootstrap_root=bootstrap,
            )
            second = daily.integrate(
                target_trade_date=date,
                data_dir=data_dir,
                output_root=output_root,
                pro=FakePro(dates),
                bootstrap_root=bootstrap,
            )

            self.assertEqual(first["events"]["target_rows"], 1)
            self.assertEqual(second["events"]["target_rows"], 1)
            self.assertEqual(second["events"]["duplicate_groups"], 0)
            self.assertEqual(second["signal"]["duplicate_groups"], 0)
            self.assertEqual(second["signal"]["future_rows"], 0)
            with duckdb.connect(
                str(output_root / "l2_stock_risk_signal.duckdb"),
                read_only=True,
            ) as connection:
                max_signal_date = connection.execute(
                    "SELECT MAX(signal_date) FROM stock_risk_signal"
                ).fetchone()[0]
            self.assertLessEqual(str(max_signal_date), date)


if __name__ == "__main__":
    unittest.main()
