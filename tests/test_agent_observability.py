import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent_observability import (
    AGENT_LABELS,
    AGENT_ORDER,
    CANONICAL_AGENT_LABELS,
    LEGACY_AGENT_LABELS,
    build_daily_snapshot,
    render_daily_markdown_report,
    write_daily_snapshot,
)


class AgentObservabilityTests(unittest.TestCase):
    def test_active_agent_catalog_matches_governed_roles(self):
        expected = {
            "architect-agent",
            "audit-agent",
            "commander-agent",
            "data-ingestion-agent",
            "data-integration-agent",
            "factor-agent",
            "mcp-agent",
            "model-agent",
            "research-agent",
            "strategy-agent",
            "trading-agent",
        }

        self.assertEqual(set(CANONICAL_AGENT_LABELS), expected)
        self.assertEqual(set(AGENT_ORDER), expected)
        self.assertEqual(len(AGENT_ORDER), len(expected))
        self.assertTrue(LEGACY_AGENT_LABELS.isdisjoint(AGENT_ORDER))
        self.assertTrue(LEGACY_AGENT_LABELS.issubset(AGENT_LABELS))

    def _set_mtime(self, path: Path, dt_text: str) -> None:
        dt = datetime.strptime(dt_text, "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        ts = dt.timestamp()
        path.touch(exist_ok=True)
        path.chmod(path.stat().st_mode)
        import os

        os.utime(path, (ts, ts))

    def _write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        path.write_text(text, encoding="utf-8")

    def test_build_daily_snapshot_prefers_exact_token_usage_and_discovers_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "quant" / "data_file"
            runtime = data_dir / "runtime" / "agent_memory"
            reports_dir = data_dir / "reports"
            prediction_dir = data_dir / "model_predictions"
            formal_manifest_dir = root / "quant" / "main" / "config" / "prediction_manifests"

            self._write_jsonl(
                runtime / "collaboration_requests.jsonl",
                [
                    {
                        "time": "2026-06-20T04:25:00+08:00",
                        "from_agent": "model-agent",
                        "summary": "发布 5d 正式 L4 资产并切换默认 manifest。",
                        "evidence": [
                            str(formal_manifest_dir / "prod_liq_prime_one_v20260612_l4_formal.json"),
                            str(reports_dir / "model_formal_release_20260620.md"),
                        ],
                    }
                ],
            )

            (reports_dir / "model_formal_release_20260620.md").parent.mkdir(parents=True, exist_ok=True)
            (reports_dir / "model_formal_release_20260620.md").write_text(
                "# 中文报告\n\n模型侧已发布正式资产。\n",
                encoding="utf-8",
            )
            self._write_json(
                formal_manifest_dir / "prod_liq_prime_one_v20260612_l4_formal.json",
                {
                    "approval_status": "approved_for_l5",
                    "table": "stock_predict_data_model_agent_toprank_latestfactor",
                },
            )
            self._write_json(
                prediction_dir / "stock_predict_data_model_agent_toprank_latestfactor" / "prediction_manifest.json",
                {
                    "approval_status": "approved_for_l5",
                    "table": "stock_predict_data_model_agent_toprank_latestfactor",
                },
            )
            self._set_mtime(
                prediction_dir / "stock_predict_data_model_agent_toprank_latestfactor" / "prediction_manifest.json",
                "20260620042000",
            )
            self._write_json(
                data_dir / "runtime" / "agent_observability" / "token_usage" / "model-agent_20260620.json",
                {
                    "snapshot_date": "20260620",
                    "agent_id": "model-agent",
                    "prompt_tokens": 1200,
                    "completion_tokens": 800,
                    "total_tokens": 2000,
                },
            )

            snapshot = build_daily_snapshot(
                snapshot_date="20260620",
                project_root=root,
                data_dir=data_dir,
            )

            model_row = next(row for row in snapshot["agents"] if row["agent_id"] == "model-agent")
            self.assertEqual(model_row["exact_total_tokens"], 2000)
            self.assertIsNone(model_row["estimated_total_tokens"])
            self.assertEqual(model_row["formal_output_count"], 2)
            self.assertGreaterEqual(model_row["report_output_count"], 1)
            self.assertGreaterEqual(model_row["business_output_count"], 3)
            self.assertEqual(model_row["token_status"], "exact")

    def test_build_daily_snapshot_estimates_tokens_when_exact_usage_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "quant" / "data_file"
            runtime = data_dir / "runtime" / "agent_memory"
            reports_dir = data_dir / "reports"

            self._write_jsonl(
                runtime / "collaboration_requests.jsonl",
                [
                    {
                        "time": "2026-06-19T09:25:20+08:00",
                        "from_agent": "factor-agent",
                        "summary": "提交行业编码治理与回收站治理证据。",
                        "evidence": [str(reports_dir / "factor_file_governance_20260619.md")],
                    }
                ],
            )
            (reports_dir / "factor_file_governance_20260619.md").parent.mkdir(parents=True, exist_ok=True)
            (reports_dir / "factor_file_governance_20260619.md").write_text(
                "# 中文治理报告\n\n这里有一段足够长的说明，用于触发估算 token。\n" * 20,
                encoding="utf-8",
            )
            self._set_mtime(reports_dir / "factor_file_governance_20260619.md", "20260619092520")

            snapshot = build_daily_snapshot(
                snapshot_date="20260619",
                project_root=root,
                data_dir=data_dir,
            )

            factor_row = next(row for row in snapshot["agents"] if row["agent_id"] == "factor-agent")
            self.assertIsNone(factor_row["exact_total_tokens"])
            self.assertGreater(factor_row["estimated_total_tokens"], 0)
            self.assertEqual(factor_row["token_status"], "estimated")
            self.assertIn("collaboration_requests", factor_row["estimate_basis"])

    def test_write_daily_snapshot_persists_duckdb_and_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "quant" / "data_file"
            runtime = data_dir / "runtime" / "agent_memory"
            reports_dir = data_dir / "reports"

            self._write_jsonl(
                runtime / "collaboration_requests.jsonl",
                [
                    {
                        "time": "2026-06-18T17:16:49+08:00",
                        "from_agent": "audit-agent",
                        "summary": "提交 GTJA 公式审计申请。",
                        "evidence": [str(reports_dir / "gtja_formula_audit_request_20260618.md")],
                    }
                ],
            )
            (reports_dir / "gtja_formula_audit_request_20260618.md").parent.mkdir(parents=True, exist_ok=True)
            (reports_dir / "gtja_formula_audit_request_20260618.md").write_text(
                "# 中文审计报告\n\nGTJA 公式审计。\n",
                encoding="utf-8",
            )
            self._set_mtime(reports_dir / "gtja_formula_audit_request_20260618.md", "20260618171649")

            result = write_daily_snapshot(
                snapshot_date="20260618",
                project_root=root,
                data_dir=data_dir,
            )

            self.assertTrue(Path(result["db_path"]).exists())
            self.assertTrue(Path(result["json_path"]).exists())
            self.assertTrue(Path(result["csv_path"]).exists())
            self.assertTrue(Path(result["md_path"]).exists())

            import duckdb

            conn = duckdb.connect(result["db_path"])
            try:
                total = conn.execute(
                    "SELECT count(*) FROM agent_daily_metrics WHERE snapshot_date = ?",
                    ("20260618",),
                ).fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(total, 1)

    def test_render_daily_markdown_report_uses_chinese_labels(self):
        snapshot = {
            "snapshot_date": "20260621",
            "generated_at": "2026-06-21T09:00:00+08:00",
            "agents": [
                {
                    "agent_id": "strategy-agent",
                    "agent_label": AGENT_LABELS["strategy-agent"],
                    "exact_total_tokens": None,
                    "estimated_total_tokens": 1234,
                    "token_status": "estimated",
                    "estimate_basis": "collaboration_requests + text_outputs",
                    "business_output_count": 4,
                    "formal_output_count": 1,
                    "report_output_count": 2,
                    "activity_count": 3,
                    "top_outputs": [
                        {
                            "relative_path": "reports/strategy_l5_remediation_execution_20260617.md",
                            "output_kind": "report_md",
                            "is_formal": 0,
                        }
                    ],
                }
            ],
        }

        text = render_daily_markdown_report(snapshot)

        self.assertIn("# 智能体产出与 Token 监测日报", text)
        self.assertIn("策略智能体", text)
        self.assertIn("估算", text)
        self.assertIn("业务产出", text)


if __name__ == "__main__":
    unittest.main()
