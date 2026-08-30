import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb

from rebuild_l2_stock_daily_duckdb_mainline import rebuild_stock_daily_duckdb_mainline


class RebuildL2StockDailyDuckdbMainlineTests(unittest.TestCase):
    def _write_table(self, path: Path, table: str, ddl: str, rows: list[tuple]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(path)) as conn:
            conn.execute(ddl)
            if rows:
                placeholders = ", ".join(["?"] * len(rows[0]))
                conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)

    def _write_active_l2_table(self, path: Path, rows: list[tuple]) -> None:
        self._write_table(
            path,
            "STOCK_DAILY_DATA",
            """
            CREATE TABLE STOCK_DAILY_DATA(
                stock_code TEXT,
                trade_date TEXT,
                open DOUBLE,
                open_qfq DOUBLE
            )
            """,
            rows,
        )

    def _write_basic_l1_pair(self, l1_dir: Path, trade_date: str, rows: list[tuple[str, str, float]]) -> None:
        self._write_table(
            l1_dir / "daily_data.duckdb",
            "daily_data",
            """
            CREATE TABLE daily_data(
                ts_code TEXT,
                trade_date TEXT,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                pre_close DOUBLE
            )
            """,
            [(code, td, open_, open_ + 1, open_ - 1, open_ + 0.5, open_ - 0.2) for code, td, open_ in rows],
        )
        self._write_table(
            l1_dir / "daily_index_data.duckdb",
            "daily_index_data",
            """
            CREATE TABLE daily_index_data(
                ts_code TEXT,
                trade_date TEXT,
                turnover_rate DOUBLE
            )
            """,
            [(code, td, 1.0) for code, td, _open in rows],
        )

    def _write_validation_baseline(self, path: Path, *, target_trade_date: str, row_count: int, stock_count: int, min_trade_date: str, max_trade_date: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "target_trade_date": target_trade_date,
                    "overall": [row_count, stock_count, min_trade_date, max_trade_date, 0],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_rebuild_uses_l1_duckdb_sources_and_excludes_bj_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "quant" / "data_file"
            l1_dir = data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
            target = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

            self._write_table(
                l1_dir / "daily_data.duckdb",
                "daily_data",
                """
                CREATE TABLE daily_data(
                    ts_code TEXT,
                    trade_date TEXT,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    pre_close DOUBLE
                )
                """,
                [
                    ("000001.SZ", "20260701", 10.0, 11.0, 9.0, 10.5, 9.8),
                    ("920001.BJ", "20260701", 20.0, 21.0, 19.0, 20.5, 19.8),
                ],
            )
            self._write_table(
                l1_dir / "adj_factor.duckdb",
                "adj_factor",
                """
                CREATE TABLE adj_factor(
                    ts_code TEXT,
                    trade_date TEXT,
                    adj_factor DOUBLE
                )
                """,
                [
                    ("000001.SZ", "20260701", 1.5),
                    ("920001.BJ", "20260701", 2.0),
                ],
            )
            self._write_table(
                l1_dir / "daily_index_data.duckdb",
                "daily_index_data",
                """
                CREATE TABLE daily_index_data(
                    ts_code TEXT,
                    trade_date TEXT,
                    turnover_rate DOUBLE
                )
                """,
                [
                    ("000001.SZ", "20260701", 1.0),
                    ("920001.BJ", "20260701", 2.0),
                ],
            )

            result = rebuild_stock_daily_duckdb_mainline(
                data_dir=data_dir,
                target_path=target,
                target_trade_date="20260701",
                raw_tables={
                    "daily_data": "daily_data",
                    "daily_index_data": "daily_index_data",
                    "adj_factor": "adj_factor",
                },
                build_sql="""
                DROP TABLE IF EXISTS STOCK_DAILY_DATA;
                CREATE TABLE STOCK_DAILY_DATA AS
                SELECT
                    T1.ts_code AS stock_code,
                    T1.trade_date,
                    T1.open AS open,
                    T1.open * T7.adj_factor AS open_qfq,
                    T1.high AS high,
                    T1.high * T7.adj_factor AS high_qfq,
                    T1.low AS low,
                    T1.low * T7.adj_factor AS low_qfq,
                    T1.close AS close,
                    T1.close * T7.adj_factor AS close_qfq,
                    T1.pre_close AS pre_close,
                    T1.pre_close * T7.adj_factor AS pre_close_qfq
                FROM daily_data AS T1
                LEFT JOIN adj_factor AS T7
                  ON T1.ts_code = T7.ts_code
                 AND T1.trade_date = T7.trade_date
                ;
                """,
                require_approved_baseline_gate=False,
                require_full_history_gate=False,
                require_snapshot_before_replace=False,
                special_code="000001.SZ",
            )

            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["active_written"])
            self.assertEqual(result["current_active_target_metrics_after"]["row_count"], 1)
            self.assertEqual(result["current_active_target_metrics_after"]["bj_row_count"], 0)
            self.assertEqual(result["target_path"], str(target))
            with duckdb.connect(str(target), read_only=True) as conn:
                rows = conn.execute(
                    """
                    SELECT stock_code, open, open_qfq, close, close_qfq, pre_close_qfq
                    FROM STOCK_DAILY_DATA
                    """
                ).fetchall()
            self.assertEqual(rows, [("000001.SZ", 10.0, 15.0, 10.5, 15.75, 14.700000000000001)])

    def test_rebuild_can_preserve_bj_when_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "quant" / "data_file"
            l1_dir = data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
            target = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

            self._write_table(
                l1_dir / "daily_data.duckdb",
                "daily_data",
                """
                CREATE TABLE daily_data(
                    ts_code TEXT,
                    trade_date TEXT,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    pre_close DOUBLE
                )
                """,
                [("920001.BJ", "20260701", 20.0, 21.0, 19.0, 20.5, 19.8)],
            )
            self._write_table(
                l1_dir / "adj_factor.duckdb",
                "adj_factor",
                """
                CREATE TABLE adj_factor(
                    ts_code TEXT,
                    trade_date TEXT,
                    adj_factor DOUBLE
                )
                """,
                [("920001.BJ", "20260701", 2.0)],
            )
            self._write_table(
                l1_dir / "daily_index_data.duckdb",
                "daily_index_data",
                """
                CREATE TABLE daily_index_data(
                    ts_code TEXT,
                    trade_date TEXT,
                    turnover_rate DOUBLE
                )
                """,
                [("920001.BJ", "20260701", 2.0)],
            )

            result = rebuild_stock_daily_duckdb_mainline(
                data_dir=data_dir,
                target_path=target,
                target_trade_date="20260701",
                raw_tables={
                    "daily_data": "daily_data",
                    "daily_index_data": "daily_index_data",
                    "adj_factor": "adj_factor",
                },
                build_sql="""
                DROP TABLE IF EXISTS STOCK_DAILY_DATA;
                CREATE TABLE STOCK_DAILY_DATA AS
                SELECT
                    T1.ts_code AS stock_code,
                    T1.trade_date,
                    T1.open AS open,
                    T1.open * T7.adj_factor AS open_qfq
                FROM daily_data AS T1
                LEFT JOIN adj_factor AS T7
                  ON T1.ts_code = T7.ts_code
                 AND T1.trade_date = T7.trade_date
                ;
                """,
                exclude_bj=False,
                require_approved_baseline_gate=False,
                require_full_history_gate=False,
                require_snapshot_before_replace=False,
                special_code="920001.BJ",
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["current_active_target_metrics_after"]["row_count"], 1)
            self.assertEqual(result["current_active_target_metrics_after"]["bj_row_count"], 1)
            with duckdb.connect(str(target), read_only=True) as conn:
                rows = conn.execute("SELECT stock_code, open_qfq FROM STOCK_DAILY_DATA").fetchall()
            self.assertEqual(rows, [("920001.BJ", 40.0)])

    def test_precheck_does_not_touch_active_target_and_writes_gate_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "quant" / "data_file"
            reports_dir = data_dir / "reports"
            l1_dir = data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
            target = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            gate_report = reports_dir / "gate_report.json"

            self._write_active_l2_table(
                target,
                [
                    ("000001.SZ", "20200101", 10.0, 10.0),
                    ("000001.SZ", "20260709", 11.0, 11.0),
                ],
            )
            reports_dir.mkdir(parents=True, exist_ok=True)
            before_mtime = target.stat().st_mtime_ns
            before_size = target.stat().st_size

            self._write_table(
                l1_dir / "daily_data.duckdb",
                "daily_data",
                """
                CREATE TABLE daily_data(
                    ts_code TEXT,
                    trade_date TEXT,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    pre_close DOUBLE
                )
                """,
                [("301583.SZ", "20260710", 10.0, 11.0, 9.0, 10.5, 9.8)],
            )
            self._write_table(
                l1_dir / "daily_index_data.duckdb",
                "daily_index_data",
                """
                CREATE TABLE daily_index_data(
                    ts_code TEXT,
                    trade_date TEXT,
                    turnover_rate DOUBLE
                )
                """,
                [("301583.SZ", "20260710", 1.0)],
            )

            result = rebuild_stock_daily_duckdb_mainline(
                data_dir=data_dir,
                target_path=target,
                target_trade_date="20260710",
                precheck_only=True,
                gate_report_json=gate_report,
                approved_baseline_min_trade_date="20200101",
                build_sql="""
                DROP TABLE IF EXISTS STOCK_DAILY_DATA;
                CREATE TABLE STOCK_DAILY_DATA AS
                SELECT
                    T1.ts_code AS stock_code,
                    T1.trade_date,
                    T1.open AS open,
                    T1.open AS open_qfq
                FROM daily_data AS T1
                JOIN daily_index_data AS T2
                  ON T1.ts_code = T2.ts_code
                 AND T1.trade_date = T2.trade_date
                ;
                """,
                raw_tables={"daily_data": "daily_data", "daily_index_data": "daily_index_data"},
            )

            self.assertEqual(result["status"], "precheck_failed")
            self.assertFalse(result["active_written"])
            self.assertTrue(result["summary"]["active_unchanged"])
            self.assertTrue(gate_report.exists())
            self.assertEqual(target.stat().st_mtime_ns, before_mtime)
            self.assertEqual(target.stat().st_size, before_size)
            with duckdb.connect(str(target), read_only=True) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM STOCK_DAILY_DATA").fetchone()[0], 2)
            source_gate = next(g for g in result["gates"] if g["name"] == "source_coverage_gate")
            self.assertFalse(source_gate["passed"])

    def test_shrink_gate_blocks_short_window_temp_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "quant" / "data_file"
            l1_dir = data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
            target = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

            self._write_active_l2_table(
                target,
                [
                    ("000001.SZ", "20200101", 10.0, 10.0),
                    ("000001.SZ", "20260709", 11.0, 11.0),
                    ("301583.SZ", "20260709", 12.0, 12.0),
                ],
            )
            self._write_table(
                l1_dir / "daily_data.duckdb",
                "daily_data",
                """
                CREATE TABLE daily_data(
                    ts_code TEXT,
                    trade_date TEXT,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    pre_close DOUBLE
                )
                """,
                [("301583.SZ", "20260710", 10.0, 11.0, 9.0, 10.5, 9.8)],
            )
            self._write_table(
                l1_dir / "daily_index_data.duckdb",
                "daily_index_data",
                """
                CREATE TABLE daily_index_data(
                    ts_code TEXT,
                    trade_date TEXT,
                    turnover_rate DOUBLE
                )
                """,
                [("301583.SZ", "20260710", 1.0)],
            )

            result = rebuild_stock_daily_duckdb_mainline(
                data_dir=data_dir,
                target_path=target,
                target_trade_date="20260710",
                precheck_only=True,
                approved_baseline_min_trade_date="20200101",
                build_sql="""
                DROP TABLE IF EXISTS STOCK_DAILY_DATA;
                CREATE TABLE STOCK_DAILY_DATA AS
                SELECT
                    T1.ts_code AS stock_code,
                    T1.trade_date,
                    T1.open AS open,
                    T1.open AS open_qfq
                FROM daily_data AS T1
                JOIN daily_index_data AS T2
                  ON T1.ts_code = T2.ts_code
                 AND T1.trade_date = T2.trade_date
                ;
                """,
                raw_tables={"daily_data": "daily_data", "daily_index_data": "daily_index_data"},
            )

            shrink_gate = next(g for g in result["gates"] if g["name"] == "shrink_gate")
            self.assertFalse(shrink_gate["passed"])
            self.assertIn("output.min_trade_date", shrink_gate["reason"])

    def test_snapshot_gate_fails_when_snapshot_path_equals_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "quant" / "data_file"
            l1_dir = data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
            target = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            self._write_active_l2_table(target, [("301583.SZ", "20260715", 10.0, 10.0)])
            self._write_basic_l1_pair(l1_dir, "20260716", [("301583.SZ", "20260716", 11.0)])

            result = rebuild_stock_daily_duckdb_mainline(
                data_dir=data_dir,
                target_path=target,
                target_trade_date="20260716",
                precheck_only=True,
                approved_baseline_min_trade_date="20260715",
                snapshot_manifest_path=str(target),
                build_sql="""
                DROP TABLE IF EXISTS STOCK_DAILY_DATA;
                CREATE TABLE STOCK_DAILY_DATA AS
                SELECT T1.ts_code AS stock_code, T1.trade_date, T1.open AS open, T1.open AS open_qfq
                FROM daily_data AS T1 JOIN daily_index_data AS T2
                  ON T1.ts_code = T2.ts_code AND T1.trade_date = T2.trade_date
                ;
                """,
                raw_tables={"daily_data": "daily_data", "daily_index_data": "daily_index_data"},
                require_approved_baseline_gate=False,
            )

            snapshot_gate = next(g for g in result["gates"] if g["name"] == "snapshot_before_replace_gate")
            self.assertFalse(snapshot_gate["passed"])
            self.assertIn("different from active target", snapshot_gate["reason"])

    def test_execute_creates_independent_snapshot_and_manifest_before_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "quant" / "data_file"
            reports_dir = data_dir / "reports"
            l1_dir = data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
            target = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            snapshot_manifest_path = reports_dir / "snapshot_manifest.json"
            baseline_path = reports_dir / "baseline_validation.json"
            self._write_active_l2_table(target, [("301583.SZ", "20260715", 10.0, 10.0)])
            before_state = target.stat().st_mtime_ns
            self._write_basic_l1_pair(l1_dir, "20260716", [("301583.SZ", "20260716", 11.0)])
            self._write_validation_baseline(
                baseline_path,
                target_trade_date="20260715",
                row_count=1,
                stock_count=1,
                min_trade_date="20260716",
                max_trade_date="20260715",
            )

            result = rebuild_stock_daily_duckdb_mainline(
                data_dir=data_dir,
                target_path=target,
                target_trade_date="20260716",
                approved_baseline_report_json=baseline_path,
                approved_baseline_min_trade_date="20260716",
                snapshot_manifest_path=snapshot_manifest_path,
                build_sql="""
                DROP TABLE IF EXISTS STOCK_DAILY_DATA;
                CREATE TABLE STOCK_DAILY_DATA AS
                SELECT T1.ts_code AS stock_code, T1.trade_date, T1.open AS open, T1.open AS open_qfq
                FROM daily_data AS T1 JOIN daily_index_data AS T2
                  ON T1.ts_code = T2.ts_code AND T1.trade_date = T2.trade_date
                ;
                """,
                raw_tables={"daily_data": "daily_data", "daily_index_data": "daily_index_data"},
                require_approved_baseline_gate=False,
            )

            snapshot_db = reports_dir / "snapshot_manifest.duckdb"
            self.assertEqual(result["status"], "completed")
            self.assertTrue(snapshot_db.exists())
            self.assertTrue(snapshot_manifest_path.exists())
            manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["asset_path"], str(snapshot_db))
            self.assertNotEqual(Path(manifest["asset_path"]).resolve(), target.resolve())
            self.assertEqual(manifest["table_metrics"]["max_trade_date"], "20260715")
            self.assertEqual(manifest["table_metrics"]["row_count"], 1)
            self.assertTrue(manifest["sha256"])
            self.assertGreater(target.stat().st_mtime_ns, before_state)
            with duckdb.connect(str(snapshot_db), read_only=True) as conn:
                self.assertEqual(conn.execute("SELECT trade_date FROM STOCK_DAILY_DATA").fetchone()[0], "20260715")
            with duckdb.connect(str(target), read_only=True) as conn:
                self.assertEqual(conn.execute("SELECT trade_date FROM STOCK_DAILY_DATA").fetchone()[0], "20260716")

    def test_snapshot_failure_blocks_replace_and_active_remains_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "quant" / "data_file"
            reports_dir = data_dir / "reports"
            l1_dir = data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
            target = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            snapshot_manifest_path = reports_dir / "snapshot_manifest.json"
            baseline_path = reports_dir / "baseline_validation.json"
            self._write_active_l2_table(target, [("301583.SZ", "20260715", 10.0, 10.0)])
            before_state = target.stat().st_mtime_ns
            self._write_basic_l1_pair(l1_dir, "20260716", [("301583.SZ", "20260716", 11.0)])
            self._write_validation_baseline(
                baseline_path,
                target_trade_date="20260715",
                row_count=1,
                stock_count=1,
                min_trade_date="20260716",
                max_trade_date="20260715",
            )

            with mock.patch("rebuild_l2_stock_daily_duckdb_mainline.shutil.copy2", side_effect=OSError("copy failed")):
                with self.assertRaises(RuntimeError):
                    rebuild_stock_daily_duckdb_mainline(
                        data_dir=data_dir,
                        target_path=target,
                        target_trade_date="20260716",
                        approved_baseline_report_json=baseline_path,
                        approved_baseline_min_trade_date="20260716",
                        snapshot_manifest_path=snapshot_manifest_path,
                        build_sql="""
                        DROP TABLE IF EXISTS STOCK_DAILY_DATA;
                        CREATE TABLE STOCK_DAILY_DATA AS
                        SELECT T1.ts_code AS stock_code, T1.trade_date, T1.open AS open, T1.open AS open_qfq
                        FROM daily_data AS T1 JOIN daily_index_data AS T2
                          ON T1.ts_code = T2.ts_code AND T1.trade_date = T2.trade_date
                        ;
                        """,
                        raw_tables={"daily_data": "daily_data", "daily_index_data": "daily_index_data"},
                        require_approved_baseline_gate=False,
                    )

            self.assertEqual(target.stat().st_mtime_ns, before_state)
            with duckdb.connect(str(target), read_only=True) as conn:
                self.assertEqual(conn.execute("SELECT trade_date FROM STOCK_DAILY_DATA").fetchone()[0], "20260715")
            self.assertFalse((reports_dir / "snapshot_manifest.duckdb").exists())

    def test_precheck_only_does_not_create_snapshot_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "quant" / "data_file"
            reports_dir = data_dir / "reports"
            l1_dir = data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
            target = data_dir / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
            snapshot_manifest_path = reports_dir / "snapshot_manifest.json"
            baseline_path = reports_dir / "baseline_validation.json"
            self._write_active_l2_table(target, [("301583.SZ", "20260715", 10.0, 10.0)])
            self._write_basic_l1_pair(l1_dir, "20260716", [("301583.SZ", "20260716", 11.0)])
            self._write_validation_baseline(
                baseline_path,
                target_trade_date="20260715",
                row_count=1,
                stock_count=1,
                min_trade_date="20260716",
                max_trade_date="20260715",
            )

            result = rebuild_stock_daily_duckdb_mainline(
                data_dir=data_dir,
                target_path=target,
                target_trade_date="20260716",
                precheck_only=True,
                approved_baseline_report_json=baseline_path,
                approved_baseline_min_trade_date="20260716",
                snapshot_manifest_path=snapshot_manifest_path,
                build_sql="""
                DROP TABLE IF EXISTS STOCK_DAILY_DATA;
                CREATE TABLE STOCK_DAILY_DATA AS
                SELECT T1.ts_code AS stock_code, T1.trade_date, T1.open AS open, T1.open AS open_qfq
                FROM daily_data AS T1 JOIN daily_index_data AS T2
                  ON T1.ts_code = T2.ts_code AND T1.trade_date = T2.trade_date
                ;
                """,
                raw_tables={"daily_data": "daily_data", "daily_index_data": "daily_index_data"},
                require_approved_baseline_gate=False,
            )

            self.assertEqual(result["status"], "precheck_passed")
            self.assertFalse(snapshot_manifest_path.exists())
            self.assertFalse((reports_dir / "snapshot_manifest.duckdb").exists())


if __name__ == "__main__":
    unittest.main()
