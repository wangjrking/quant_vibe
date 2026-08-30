from __future__ import annotations

import json
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
STRATEGY_LIBRARY_DIR = MAIN_DIR / "strategy_library"
REGISTRY_PATH = STRATEGY_LIBRARY_DIR / "registry.json"
PRODUCTION_ASSET_REGISTRY_PATH = MAIN_DIR.parent / "data_file" / "asset_registry" / "production_assets.json"
DATA_FILE_DIR = MAIN_DIR.parent / "data_file"
ACTIVE_L2_DUCKDB = (
    MAIN_DIR.parent
    / "data_file"
    / "production_assets"
    / "duckdb"
    / "l2_stock_daily_data.duckdb"
)
FORBIDDEN_CURRENT_ROUTE_TOKENS = (
    "MODEL_PREDICTIONS.db",
    "STOCK_DAILY_DATA.db",
    "quant_production.duckdb",
    "odb.db",
)
ACTIVE_DUCKDB_ROOT_TOKEN = "quant/data_file/production_assets/duckdb/"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class StrategyLibraryCurrentAndHistoricalContractsTests(unittest.TestCase):
    def test_data_file_root_does_not_keep_legacy_shap_artifacts(self):
        legacy_root_files = [
            DATA_FILE_DIR / "shap_values.pkl",
            DATA_FILE_DIR / "test_index.pkl",
        ]
        failures = [str(path) for path in legacy_root_files if path.exists()]
        if failures:
            self.fail(
                "legacy shap artifacts must not remain in data_file root:\n"
                + "\n".join(failures)
            )

    def test_runtime_root_does_not_keep_archived_loose_history_files(self):
        runtime_dir = DATA_FILE_DIR / "runtime"
        forbidden_root_files = [
            runtime_dir / "qmt_screen.png",
            runtime_dir / "qmt_screen_2.png",
            runtime_dir / "orchestrator_task_assignments_20260616.json",
            runtime_dir / "tmp_signal_probe_20260629.csv",
            runtime_dir / "tmp_signal_probe_20260629_status.json",
            runtime_dir / "candidate_move_root_old_scripts_20260701.json",
            runtime_dir / "candidate_move_root_old_scripts_round2_20260701.json",
            runtime_dir / "candidate_move_root_old_scripts_round3_20260701.json",
        ]
        failures = [str(path) for path in forbidden_root_files if path.exists()]
        if failures:
            self.fail(
                "runtime root must not keep archived or recycled loose history files:\n"
                + "\n".join(failures)
            )

    def test_reports_root_does_not_keep_recycled_history_result_directories(self):
        reports_dir = DATA_FILE_DIR / "reports"
        forbidden_report_dirs = [
            reports_dir / "strategy_agent_next_open_opt_ce10_grid_20260701",
            reports_dir / "strategy_agent_next_open_opt_ce10_20260701",
            reports_dir / "strategy_agent_next_open_opt_ce10_peak_20260701",
            reports_dir / "strategy_agent_next_open_opt_ce10_peak2_20260701",
            reports_dir / "strategy_agent_next_open_opt_ce10_retry_20260701",
            reports_dir / "strategy_agent_next_open_exec_focus_20260701",
            reports_dir / "strategy_agent_next_open_opt_latest_l4_top3_20260701",
            reports_dir / "strategy_agent_next_open_refill_latest_l4_top3_20260701",
            reports_dir / "strategy_agent_open_only_corrected_frequency_search_20260701",
            reports_dir / "strategy_agent_sellfreq_continue_u9d9_20260701",
            reports_dir / "strategy_agent_top1_open_gap_refill_20260701",
            reports_dir / "strategy_agent_sellfreq_u9d9_20260701",
            reports_dir / "strategy_agent_latest_l4_direction_diagnostic_20260702",
            reports_dir / "strategy_agent_latest_l4_open_gap_execution_probe_20260702",
            reports_dir / "strategy_agent_latest_l4_20260701_top35_frequency_grid_20260702",
            reports_dir / "strategy_agent_latest_l4_next_open_weight_search_20260702",
            reports_dir / "strategy_agent_next_open_weight_candidate_20260702",
            reports_dir / "strategy_agent_recent_two_week_signals_20260702",
            reports_dir / "model_agent_l4_standby_no_bj_duckdb_only_20260701",
        ]
        failures = [str(path) for path in forbidden_report_dirs if path.exists()]
        if failures:
            self.fail(
                "reports root must not keep recycled historical result directories:\n"
                + "\n".join(failures)
            )

    def test_data_file_has_no_non_recycle_sqlite_like_files(self):
        failures: list[str] = []
        recycle_root = (DATA_FILE_DIR / "runtime" / "recycle_bin").resolve()

        for path in DATA_FILE_DIR.rglob("*"):
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path

            try:
                resolved.relative_to(recycle_root)
                continue
            except ValueError:
                pass

            name = path.name.lower()
            suffix = path.suffix.lower()
            if suffix in {".db", ".sqlite", ".sqlite3"} or name.endswith(".db"):
                failures.append(str(path))

        if failures:
            self.fail(
                "non-recycle sqlite-like files remain under data_file:\n"
                + "\n".join(sorted(failures))
            )

    def test_data_file_has_no_non_recycle_shared_quant_production_duckdb(self):
        failures: list[str] = []
        recycle_root = (DATA_FILE_DIR / "runtime" / "recycle_bin").resolve()

        for path in DATA_FILE_DIR.rglob("quant_production.duckdb"):
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path

            try:
                resolved.relative_to(recycle_root)
                continue
            except ValueError:
                pass

            failures.append(str(path))

        if failures:
            self.fail(
                "shared quant_production.duckdb must not remain outside recycle_bin:\n"
                + "\n".join(sorted(failures))
            )

    def test_active_production_asset_registry_is_duckdb_only_for_current_routes(self):
        registry = _load_json(PRODUCTION_ASSET_REGISTRY_PATH)
        failures: list[str] = []

        for asset in registry.get("assets", []):
            if not asset.get("allowed_for_main_workflow"):
                continue

            asset_id = str(asset.get("asset_id") or "")
            asset_type = str(asset.get("asset_type") or "")
            asset_path = str(asset.get("asset_path") or "")
            resolved_db_path = str(asset.get("resolved_db_path") or "")

            if asset_type == "duckdb_table":
                if not asset_path.startswith(ACTIVE_DUCKDB_ROOT_TOKEN):
                    failures.append(f"{asset_id}: duckdb_table asset_path must stay under production_assets/duckdb")
                if ".duckdb::" not in asset_path:
                    failures.append(f"{asset_id}: duckdb_table asset_path must include .duckdb::table binding")

            if asset_type == "duckdb_table_files":
                if not asset_path.startswith(ACTIVE_DUCKDB_ROOT_TOKEN):
                    failures.append(f"{asset_id}: duckdb_table_files asset_path must stay under production_assets/duckdb")

            if asset_type == "formal_prediction_manifest":
                if not asset_path.endswith(".json"):
                    failures.append(f"{asset_id}: formal_prediction_manifest asset_path must point to manifest json")
                if not resolved_db_path.lower().endswith(".duckdb"):
                    failures.append(f"{asset_id}: formal_prediction_manifest resolved_db_path must point to split DuckDB")

            for token in FORBIDDEN_CURRENT_ROUTE_TOKENS:
                if token in asset_path:
                    failures.append(f"{asset_id}: asset_path must not retain {token}")
                if resolved_db_path and token in resolved_db_path:
                    failures.append(f"{asset_id}: resolved_db_path must not retain {token}")

        if failures:
            self.fail("\n".join(failures))

    def test_current_production_strategy_manifest_uses_split_duckdb_only_contract(self):
        registry = _load_json(REGISTRY_PATH)
        current_strategy_id = registry["production"]["current"]
        current_entry = None
        for entry in registry["production"]["strategies"]:
            if entry.get("strategy_id") == current_strategy_id:
                current_entry = entry
                break
        self.assertIsNotNone(current_entry, "current production strategy must exist in registry")

        strategy_dir = MAIN_DIR / current_entry["path"]
        manifest_path = strategy_dir / "strategy_manifest.json"
        trading_rules_path = strategy_dir / "trading_rules.json"
        self.assertTrue(manifest_path.exists(), f"missing current strategy manifest: {manifest_path}")
        self.assertTrue(trading_rules_path.exists(), f"missing current trading rules: {trading_rules_path}")

        manifest = _load_json(manifest_path)
        trading_rules = _load_json(trading_rules_path)
        manifest_text = manifest_path.read_text(encoding="utf-8")

        self.assertEqual(manifest.get("status"), "production")
        self.assertEqual(
            str(manifest["input_contract"]["market_db_path"]).lower(),
            str(ACTIVE_L2_DUCKDB).lower(),
        )
        self.assertFalse(manifest["input_contract"].get("allow_legacy", True))
        self.assertEqual(trading_rules.get("status"), "production")
        self.assertTrue(trading_rules.get("selection_rule", {}).get("exclude_bj"))

        for token in FORBIDDEN_CURRENT_ROUTE_TOKENS:
            self.assertNotIn(token, manifest_text, f"current strategy manifest must not retain {token}")

    def test_current_production_strategy_package_and_latest_status_do_not_retain_legacy_db_tokens(self):
        registry = _load_json(REGISTRY_PATH)
        current_strategy_id = registry["production"]["current"]
        current_entry = None
        for entry in registry["production"]["strategies"]:
            if entry.get("strategy_id") == current_strategy_id:
                current_entry = entry
                break
        self.assertIsNotNone(current_entry, "current production strategy must exist in registry")

        strategy_dir = MAIN_DIR / current_entry["path"]
        prediction_meta_path = strategy_dir / "prediction_table_meta.json"
        latest_status_path = (
            MAIN_DIR.parent
            / "data_file"
            / "production_signals"
            / f"{current_strategy_id}_latest_status.json"
        )

        self.assertTrue(prediction_meta_path.exists(), f"missing current prediction metadata: {prediction_meta_path}")
        self.assertTrue(latest_status_path.exists(), f"missing current latest status: {latest_status_path}")

        prediction_meta = _load_json(prediction_meta_path)
        latest_status = _load_json(latest_status_path)
        prediction_meta_text = prediction_meta_path.read_text(encoding="utf-8")
        latest_status_text = latest_status_path.read_text(encoding="utf-8")

        for label_key in ("3d", "5d", "10d"):
            asset = prediction_meta["formal_l4_assets"][label_key]
            self.assertEqual(asset.get("source_type"), "duckdb_table")
            self.assertTrue(str(asset.get("db_path") or "").lower().endswith(".duckdb"))
            self.assertNotIn("quant_production.duckdb", str(asset.get("db_path") or ""))

        self.assertEqual(latest_status.get("strategy_id"), current_strategy_id)
        self.assertEqual(latest_status.get("status"), "pending_buy_day_hard_gate")

        for token in FORBIDDEN_CURRENT_ROUTE_TOKENS:
            self.assertNotIn(token, prediction_meta_text, f"current prediction metadata must not retain {token}")
            self.assertNotIn(token, latest_status_text, f"current latest status must not retain {token}")

    def test_current_strategy_reports_match_latest_signal_dates_and_do_not_refer_legacy_db(self):
        registry = _load_json(REGISTRY_PATH)
        current_strategy_id = registry["production"]["current"]
        current_entry = None
        for entry in registry["production"]["strategies"]:
            if entry.get("strategy_id") == current_strategy_id:
                current_entry = entry
                break
        self.assertIsNotNone(current_entry, "current production strategy must exist in registry")

        strategy_dir = MAIN_DIR / current_entry["path"]
        latest_status_path = (
            MAIN_DIR.parent
            / "data_file"
            / "production_signals"
            / f"{current_strategy_id}_latest_status.json"
        )
        latest_status = _load_json(latest_status_path)
        expected_signal_date = str(latest_status.get("signal_date"))
        expected_buy_date = str(latest_status.get("buy_date"))

        for report_name in ("report.md", "strategy_report.md"):
            report_path = strategy_dir / report_name
            self.assertTrue(report_path.exists(), f"missing current report: {report_path}")
            text = report_path.read_text(encoding="utf-8")
            self.assertIn(expected_signal_date, text, f"{report_path}: must reflect latest signal_date")
            self.assertIn(expected_buy_date, text, f"{report_path}: must reflect latest buy_date")
            self.assertIn("pending_buy_day_hard_gate", text, f"{report_path}: must reflect current gate state")
            for token in FORBIDDEN_CURRENT_ROUTE_TOKENS:
                self.assertNotIn(token, text, f"{report_path}: current report must not retain {token}")

    def test_current_l7_delivery_package_matches_pending_gate_semantics_and_split_l2_route(self):
        registry = _load_json(REGISTRY_PATH)
        current_strategy_id = registry["production"]["current"]
        latest_status_path = (
            MAIN_DIR.parent
            / "data_file"
            / "production_signals"
            / f"{current_strategy_id}_latest_status.json"
        )
        latest_status = _load_json(latest_status_path)
        signal_date = str(latest_status.get("signal_date"))
        buy_date = str(latest_status.get("buy_date"))
        delivery_dir = (
            MAIN_DIR.parent
            / "data_file"
            / "runtime"
            / "trading_agent"
            / "delivery_packages"
            / f"{current_strategy_id}_{signal_date}_for_{buy_date}"
        )
        summary_path = delivery_dir / "l7_delivery_summary.json"
        hard_gate_summary_path = delivery_dir / "buy_day_hard_gate_current_check" / "buy_day_hard_gate_summary.json"

        self.assertTrue(summary_path.exists(), f"missing current L7 delivery summary: {summary_path}")
        self.assertTrue(hard_gate_summary_path.exists(), f"missing current L7 hard-gate summary: {hard_gate_summary_path}")

        summary = _load_json(summary_path)
        hard_gate_summary = _load_json(hard_gate_summary_path)
        summary_text = summary_path.read_text(encoding="utf-8")
        hard_gate_text = hard_gate_summary_path.read_text(encoding="utf-8")

        self.assertEqual(summary.get("strategy_id"), current_strategy_id)
        self.assertEqual(summary.get("signal_date"), signal_date)
        self.assertEqual(summary.get("buy_date"), buy_date)
        self.assertEqual(summary.get("latest_status", {}).get("status"), "pending_buy_day_hard_gate")
        self.assertEqual(hard_gate_summary.get("status"), "pending_buy_day_hard_gate")
        self.assertEqual(str(hard_gate_summary.get("market_db_path")).lower(), str(ACTIVE_L2_DUCKDB).lower())

        for token in FORBIDDEN_CURRENT_ROUTE_TOKENS:
            self.assertNotIn(token, summary_text, f"{summary_path}: current L7 delivery summary must not retain {token}")
            self.assertNotIn(token, hard_gate_text, f"{hard_gate_summary_path}: current L7 hard-gate summary must not retain {token}")

    def test_successful_l4_incremental_report_dirs_do_not_keep_superseded_blocked_artifacts(self):
        report_roots = sorted(DATA_FILE_DIR.glob("reports/model_agent_formal_incremental_l4_*"))
        failures: list[str] = []

        for report_root in report_roots:
            if not report_root.is_dir():
                continue
            success_reports = list(report_root.glob("formal_incremental_l4_*_report.json"))
            if not success_reports:
                continue
            blocked_reports = list(report_root.glob("formal_incremental_l4_*_blocked.json"))
            blocked_markdown = list(report_root.glob("formal_incremental_l4_*_blocked.md"))
            qfq_fix_summaries = list(report_root.glob("qfq_contract_fix_summary_*.md"))

            for path in blocked_reports + blocked_markdown + qfq_fix_summaries:
                failures.append(str(path))

        if failures:
            self.fail(
                "successful L4 incremental report dirs must not keep superseded blocked/fix-summary artifacts:\n"
                + "\n".join(sorted(failures))
            )

    def test_historical_retained_strategy_manifests_are_explicitly_non_current(self):
        registry = _load_json(REGISTRY_PATH)
        failures: list[str] = []

        for retained in registry.get("historical_production", {}).get("retained_directories", []):
            strategy_dir = MAIN_DIR / retained["path"]
            manifest_path = strategy_dir / "strategy_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = _load_json(manifest_path)

            if manifest.get("status") == "production":
                failures.append(f"{manifest_path}: historical retained manifest must not keep status=production")
            if manifest.get("allowed_for_main_workflow") is not False:
                failures.append(f"{manifest_path}: allowed_for_main_workflow must be false")
            if manifest.get("current_route_eligible") is not False:
                failures.append(f"{manifest_path}: current_route_eligible must be false")
            if manifest.get("reference_role") != "historical_reproduction_only":
                failures.append(f"{manifest_path}: reference_role must be historical_reproduction_only")

            governance = manifest.get("governance") or {}
            if governance.get("is_current_l5") is True:
                failures.append(f"{manifest_path}: governance.is_current_l5 must be false")
            if governance.get("is_only_registered_l5_production_strategy") is True:
                failures.append(
                    f"{manifest_path}: governance.is_only_registered_l5_production_strategy must be false"
                )
            if governance.get("production") is True:
                failures.append(f"{manifest_path}: governance.production must be false")

        if failures:
            self.fail("\n".join(failures))

    def test_historical_prediction_metadata_keeps_historical_source_types(self):
        registry = _load_json(REGISTRY_PATH)
        failures: list[str] = []

        for retained in registry.get("historical_production", {}).get("retained_directories", []):
            strategy_dir = MAIN_DIR / retained["path"]
            prediction_meta_path = strategy_dir / "prediction_table_meta.json"
            if not prediction_meta_path.exists():
                continue
            text = prediction_meta_path.read_text(encoding="utf-8")
            if '"source_type": "sqlite_table"' in text:
                failures.append(f"{prediction_meta_path}: historical metadata must not expose sqlite_table as current")
            if '"source_type": "duckdb_table"' in text:
                failures.append(f"{prediction_meta_path}: historical metadata must not expose duckdb_table as current")
            if "historical_" not in text:
                failures.append(f"{prediction_meta_path}: historical metadata should keep explicit historical source_type")

        if failures:
            self.fail("\n".join(failures))


if __name__ == "__main__":
    unittest.main()
