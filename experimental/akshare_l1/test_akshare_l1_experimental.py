from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import duckdb
import pandas as pd

import akshare_l1_experimental as module


class ExperimentalAkshareL1Tests(unittest.TestCase):
    def test_production_output_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            module.validate_output_root(module.PRODUCTION_ROOT / "forbidden")

    def test_news_transform_has_stable_key(self) -> None:
        source = pd.DataFrame(
            {
                "关键词": ["样本", "样本"],
                "新闻标题": ["标题", "标题"],
                "新闻内容": ["内容", "内容"],
                "发布时间": ["2026-07-13 10:00:00", "2026-07-13 10:00:00"],
                "文章来源": ["来源", "来源"],
                "新闻链接": ["https://example.test/1", "https://example.test/1"],
            }
        )
        result = module.transform_stock_news(source, "688006.SH", "2026-07-14T10:00:00+08:00")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ts_code"], "688006.SH")
        self.assertEqual(module.duplicate_groups(result, ["news_id"]), 0)

    def test_financial_transform_preserves_legitimate_nulls(self) -> None:
        source = pd.DataFrame(
            {
                "SECUCODE": ["000001.SZ"],
                "REPORT_DATE": ["2026-03-31"],
                "GROSS_PROFIT": [None],
            }
        )
        result = module.transform_financial_indicator(source)
        self.assertEqual(result.iloc[0]["report_date"], "20260331")
        self.assertTrue(pd.isna(result.iloc[0]["GROSS_PROFIT"]))
        self.assertNotIn("REPORT_DATE", result.columns)
        self.assertEqual([column for column in result.columns if column.lower() == "report_date"], ["report_date"])

    def test_social_sentiment_remains_name_only(self) -> None:
        result = module.transform_social_sentiment(pd.DataFrame({"name": ["比亚迪"], "rate": [-0.75]}), "2026-07-14T10:00:00+08:00")
        self.assertNotIn("ts_code", result.columns)
        self.assertEqual(result.iloc[0]["mapping_status"], "partial_name_only_no_ts_code")

    def test_risk_warning_requires_explicit_code(self) -> None:
        with self.assertRaises(KeyError):
            module.transform_risk_warning(pd.DataFrame({"名称": ["ST样本"]}), "20260713")

    def test_minute_transform_supports_sina_schema(self) -> None:
        source = pd.DataFrame(
            {
                "day": ["2026-07-13 09:31:00"],
                "open": [10.0],
                "high": [10.1],
                "low": [9.9],
                "close": [10.05],
                "volume": [1000],
            }
        )
        result = module.transform_minutes(source, "000001.SZ")
        self.assertEqual(result.iloc[0]["ts_code"], "000001.SZ")
        self.assertEqual(result.iloc[0]["source"], "AKShare.stock_zh_a_minute")
        self.assertEqual(module.duplicate_groups(result, ["ts_code", "trade_time"]), 0)

    def test_one_table_one_duckdb(self) -> None:
        with tempfile.TemporaryDirectory(dir=module.EXPERIMENTAL_ROOT) as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({"id": [1], "value": ["x"]})
            path = module.write_one_table_duckdb(root, "sample_table", frame)
            connection = duckdb.connect(str(path), read_only=True)
            try:
                self.assertEqual(connection.execute("SHOW TABLES").fetchall(), [("sample_table",)])
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM sample_table").fetchone()[0], 1)
            finally:
                connection.close()


class RemediatedArtifactContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = module.DEFAULT_OUTPUT_ROOT
        cls.manifest = json.loads((cls.root / "manifest.json").read_text(encoding="utf-8"))

    def test_all_manifest_assets_have_explicit_contract(self) -> None:
        self.assertEqual(set(self.manifest["assets"]), set(module.ASSET_CONTRACTS))
        for name, asset in self.manifest["assets"].items():
            self.assertEqual(asset["table"], name)
            self.assertTrue(asset["source"])
            self.assertTrue(asset["status"].startswith(("experimental/test", "blocked_")))
            self.assertTrue(asset["not_approved_for_production"])
            self.assertFalse(asset["production_approved"])
            self.assertTrue(asset["one_table_one_file"])

    def test_manifest_table_matches_unique_physical_table(self) -> None:
        for name, asset in self.manifest["assets"].items():
            path = Path(asset["db_path"])
            connection = duckdb.connect(str(path), read_only=True)
            try:
                tables = connection.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='main' AND table_type='BASE TABLE' ORDER BY table_name"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(tables, [(name,)])

    def test_financial_schema_is_canonical(self) -> None:
        asset = self.manifest["assets"]["financial_indicator"]
        connection = duckdb.connect(asset["db_path"], read_only=True)
        try:
            columns = [row[1] for row in connection.execute("PRAGMA table_info('financial_indicator')").fetchall()]
        finally:
            connection.close()
        self.assertIn("report_date", columns)
        self.assertNotIn("REPORT_DATE_1", columns)
        self.assertEqual([column for column in columns if column.lower() == "report_date"], ["report_date"])
        self.assertEqual(asset["natural_key"], ["ts_code", "report_date"])
        self.assertEqual(asset["date_field"], "report_date")

    def test_minute_lineage_matches_actual_fallback_call(self) -> None:
        asset = self.manifest["assets"]["stock_minutes_1m"]
        self.assertEqual(asset["source"], "AKShare.stock_zh_a_minute")
        self.assertEqual(asset["fallback_from"], "AKShare.stock_zh_a_hist_min_em")
        self.assertEqual(asset["source_parameters"]["period"], "1")
        self.assertEqual(asset["source_parameters"]["adjust"], "")
        self.assertEqual(asset["adjustment"]["semantics"], "unadjusted/raw_price")
        self.assertEqual(asset["sample_scope"]["codes"], ["603026.SH", "688006.SH", "301338.SZ"])
        self.assertEqual(asset["sample_scope"]["trade_date"], "20260714")
        connection = duckdb.connect(asset["db_path"], read_only=True)
        try:
            sources = connection.execute("SELECT source, COUNT(*) FROM stock_minutes_1m GROUP BY source").fetchall()
        finally:
            connection.close()
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0][0], "AKShare.stock_zh_a_minute")
        self.assertGreater(sources[0][1], 0)

    def test_outputs_cannot_target_production_or_legacy_routes(self) -> None:
        for asset in self.manifest["assets"].values():
            path = Path(asset["db_path"]).resolve()
            self.assertTrue(module._is_relative_to(path, module.EXPERIMENTAL_ROOT.resolve()))
            self.assertNotIn("production_assets", path.parts)
            self.assertNotIn("odb.db", str(path).lower())
        self.assertFalse(self.manifest["production_assets_touched"])
        self.assertFalse(self.manifest["production_route_or_registry_touched"])
        self.assertFalse(self.manifest["l2_l8_triggered"])
        self.assertFalse(self.manifest["legacy_odb_used"])

    def test_five_assets_have_zero_duplicate_groups(self) -> None:
        duckdb_files = sorted(self.root.glob("*/*.duckdb"))
        self.assertEqual(len(duckdb_files), 5)
        for name, asset in self.manifest["assets"].items():
            self.assertEqual(asset["duplicate_key_groups"], 0, name)
            self.assertEqual(Path(asset["db_path"]), self.root / name / f"{name}.duckdb")


if __name__ == "__main__":
    unittest.main()
