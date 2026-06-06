import sqlite3
import tempfile
import unittest
from pathlib import Path

from export_gm_limit_signals import build_limit_signals


class ExportGmLimitSignalsTests(unittest.TestCase):
    def test_build_limit_signals_uses_next_daily_trade_date_and_ranks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE "daily_data" (trade_date TEXT)')
            conn.executemany('INSERT INTO "daily_data" VALUES (?)', [("20260102",), ("20260105",)])
            conn.execute(
                """
                CREATE TABLE "limit_list_data" (
                    trade_date TEXT, ts_code TEXT, industry TEXT, name TEXT, close REAL,
                    pct_chg REAL, amount REAL, limit_amount TEXT, float_mv REAL, total_mv REAL,
                    turnover_ratio REAL, fd_amount REAL, first_time TEXT, last_time TEXT,
                    open_times TEXT, up_stat TEXT, limit_times TEXT, "limit" TEXT
                )
                """
            )
            conn.executemany(
                'INSERT INTO "limit_list_data" VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                [
                    ("20260102", "600000.SH", "bank", "A", 10, 10, 100, "None", 1, 1, 1, 200, "093100", "145000", "1", "1/1", "1", "U"),
                    ("20260102", "000001.SZ", "bank", "B", 10, 10, 200, "None", 1, 1, 1, 300, "093000", "145000", "2", "1/1", "1", "U"),
                ],
            )
            conn.commit()
            conn.close()

            signals = build_limit_signals(db, "20260102", "20260105", top_k=1)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["buy_date"], "20260105")
        self.assertEqual(signals[0]["symbol"], "SHSE.600000")


if __name__ == "__main__":
    unittest.main()
