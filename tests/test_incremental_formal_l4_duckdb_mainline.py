import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjustment_semantics import default_adjustment_semantics, default_market_field_semantics
import incremental_formal_l4_duckdb_mainline as target


class IncrementalFormalL4DuckdbMainlineTests(unittest.TestCase):
    def test_resolve_savedmodel_feature_alias_maps_legacy_qfq_names(self):
        available = {
            "stock_code",
            "trade_date",
            "close_qfq",
            "macdsignal_qfq",
            "gtja_alpha024_qfq",
            "amount",
        }

        resolved = target.resolve_savedmodel_feature_aliases(
            ["stock_code", "trade_date", "close", "macdsignal", "gtja_alpha024", "amount"],
            available,
        )

        self.assertEqual(
            resolved["query_columns"],
            ["stock_code", "trade_date", "close_qfq", "macdsignal_qfq", "gtja_alpha024_qfq", "amount"],
        )
        self.assertEqual(
            resolved["alias_pairs"],
            [
                ("close", "close_qfq"),
                ("macdsignal", "macdsignal_qfq"),
                ("gtja_alpha024", "gtja_alpha024_qfq"),
            ],
        )
        self.assertEqual(resolved["missing"], [])

    def test_resolve_savedmodel_feature_alias_does_not_map_unrelated_field(self):
        available = {"stock_code", "trade_date", "amount_qfq"}

        resolved = target.resolve_savedmodel_feature_aliases(
            ["stock_code", "trade_date", "amount"],
            available,
        )

        self.assertEqual(resolved["query_columns"], ["stock_code", "trade_date"])
        self.assertEqual(resolved["alias_pairs"], [])
        self.assertEqual(resolved["missing"], ["amount"])

    def test_manifest_formula_info_defaults_to_savedmodel_completion_only(self):
        info = target.manifest_formula_info({}, factor_input="duckdb::factor", label_input="duckdb::label")

        self.assertEqual(info["formula"], "latest_incremental_rows: score = savedmodel_completion")
        self.assertEqual(info["factor_input"], "duckdb::factor")
        self.assertEqual(info["label_input"], "duckdb::label")
        self.assertEqual(info["sources"], ["savedmodel_completion"])
        self.assertEqual(info["factor_input_source_type"], "duckdb_table")
        self.assertEqual(info["factor_input_role"], "active")
        self.assertIsNone(info["partial_exception"])

    def test_duckdb_input_provenance_marks_workspace_candidate(self):
        candidate_path = (
            target.ROOT
            / "quant/data_file/runtime/agent_workspaces/factor-agent/work/run"
            / "l3_feature_candidate_20260824.duckdb"
        )

        provenance = target.duckdb_input_provenance(candidate_path, "factor_table")

        self.assertEqual(provenance["source_type"], "candidate_duckdb_table")
        self.assertEqual(provenance["input_role"], "candidate_only")
        self.assertTrue(provenance["asset"].endswith("::factor_table"))

    def test_manifest_formula_info_ignores_legacy_bestset_formula(self):
        info = target.manifest_formula_info(
            {
                "score_formula": "score = bestset_score when available; otherwise savedmodel_completion",
                "formula_sources": ["legacy_bestset"],
            },
            factor_input="duckdb::factor",
            label_input="duckdb::label",
        )

        self.assertEqual(info["formula"], "latest_incremental_rows: score = savedmodel_completion")
        self.assertEqual(info["sources"], ["savedmodel_completion"])

    def test_build_formal_frame_marks_savedmodel_completion_without_sqlite_overlay(self):
        scored = pd.DataFrame(
            {
                "trade_date": ["20260630"],
                "stock_code": ["000001.SZ"],
                "pred_prob": [0.42],
            }
        )
        label_frame = pd.DataFrame(
            {
                "trade_date": ["20260630"],
                "stock_code": ["000001.SZ"],
                "executable_5d_open_return": [None],
            }
        )

        result = target.build_formal_frame(
            trade_date="20260630",
            label_column="executable_5d_open_return",
            scored=scored,
            label_frame=label_frame,
        )

        self.assertEqual(result.loc[0, "pred_prob"], 0.42)
        self.assertEqual(result.loc[0, "score_source"], "savedmodel_completion")

    def test_build_manifest_update_payload_keeps_duckdb_contract(self):
        stats = {
            "row_count": 10,
            "trade_days": 2,
            "stock_count": 8,
            "min_trade_date": "20240604",
            "max_trade_date": "20260626",
            "latest_day_rows": 5,
            "latest_day_stock_count": 5,
            "duplicate_key_groups": 0,
            "null_pred_prob": 0,
            "pred_prob_sha256": "abc123",
        }
        formula_info = {
            "formula": "score = base",
            "sources": ["a", "b"],
            "factor_input": "duckdb::factor",
            "label_input": "duckdb::label",
        }

        payload = target.build_manifest_update_payload(
            manifest={
                "score_source_usage": {"bestset_score": {"row_count": 9}},
                "completion_row_share": 0.1,
                "lineage_sources": {"best_sqlite_table": "legacy_table"},
            },
            label_key="5d",
            stats=stats,
            formula_info=formula_info,
            archive_name="archive.json",
            target_date="20260626",
            db_path_in_manifest=target.formal_duckdb_manifest_path("5d"),
        )

        self.assertEqual(payload["source_type"], "duckdb_table")
        self.assertTrue(str(payload["db_path"]).endswith("l4_executable_5d_open_return_formal.duckdb"))
        self.assertTrue(str(payload["market_db_path"]).endswith("l2_stock_daily_data.duckdb"))
        self.assertEqual(payload["table"], target.FORMAL_TABLES["5d"])
        self.assertEqual(payload["adjustment_semantics"], default_adjustment_semantics())
        self.assertEqual(payload["market_field_semantics"], default_market_field_semantics())
        self.assertEqual(payload["latest_incremental_update"]["trade_date"], "20260626")
        self.assertNotIn("score_source_usage", payload)
        self.assertNotIn("completion_row_share", payload)
        self.assertEqual(payload["lineage"], "current_incremental_duckdb_only_savedmodel_completion")
        self.assertEqual(payload["lineage_sources"]["current_score_source"], "savedmodel_completion")
        self.assertEqual(payload["lineage_sources"]["historical_release_lineage_archive"], "archive.json")
        self.assertEqual(
            payload["latest_incremental_update"]["report_path"],
            "../../../data_file/reports/model_agent_formal_incremental_l4_20260626/formal_incremental_l4_20260626_report.json",
        )


if __name__ == "__main__":
    unittest.main()
