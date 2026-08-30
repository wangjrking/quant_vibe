"""Build an auditable L4-to-L5 handoff from readback evidence only.

This utility never opens prediction assets for writing.  It turns the formal
incremental report, canonical digest readback and approved manifests into a
single JSON contract that an independent auditor can parse and replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "quant/main/config/prediction_manifests"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--l3-handoff", required=True)
    parser.add_argument(
        "--agent-model-route",
        required=True,
        help="Actual agent runtime route recorded for this incremental run.",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir).resolve()
    target_date = str(args.target_date)
    report_path = report_dir / f"formal_incremental_l4_{target_date}_report.json"
    validation_path = report_dir / f"formal_incremental_l4_{target_date}_postwrite_validation.json"
    l3_handoff_path = Path(args.l3_handoff).resolve()
    report = load(report_path)
    validation = load(validation_path)
    l3_handoff = load(l3_handoff_path)
    factor_input_provenance = report["inputs"].get(
        "factor_input_provenance",
        {
            "asset": report["inputs"]["factor_asset"],
            "source_type": "duckdb_table",
            "input_role": "active",
        },
    )
    partial_exception = report["inputs"].get("partial_exception")

    outputs = validation["outputs"]
    output_assets: list[str] = []
    manifest_bindings: list[dict[str, Any]] = []
    errors: list[str] = []
    for manifest_name, probe in outputs.items():
        manifest_path = (MANIFEST_DIR / manifest_name).resolve()
        manifest = load(manifest_path)
        output_assets.append(f"{probe['db_path']}::{probe['table']}")
        manifest_bindings.append(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "db_path": probe["db_path"],
                "table": probe["table"],
                "source_type": manifest.get("source_type"),
                "approval_status": manifest.get("approval_status"),
                "production_model_asset_id": manifest.get("production_model_asset_id"),
                "canonical_target_day_digest_sha256": probe["canonical_digest_sha256"],
                "canonical_target_day_digest_algorithm": probe["canonical_digest_algorithm"],
            }
        )
        if not probe.get("valid"):
            errors.append(f"target-day probe failed: {manifest_name}")
        if manifest.get("source_type") != "duckdb_table":
            errors.append(f"non-DuckDB manifest: {manifest_name}")
        if manifest.get("approval_status") != "approved_for_l5":
            errors.append(f"non-approved manifest: {manifest_name}")

    expected_rows = report["inputs"]["factor_status"]["factor_target_rows"]
    expected_stocks = report["inputs"]["factor_status"]["factor_target_stocks"]
    if validation.get("all4_common_key_count") != expected_stocks:
        errors.append("all4 common key domain does not match active L3 target stocks")
    if not validation.get("valid"):
        errors.append("postwrite validation is not valid")

    contract = {
        "schema_version": 2,
        "contract_type": "workflow_layer_handoff",
        "workflow_template": "standard_incremental_trading_signal_l1_l8",
        "workflow_run_id": f"incremental-trading-signal-{target_date}-L4-formal-prediction-refresh",
        "layer": "L4",
        "owner_agent": "model-agent",
        "target_trade_date": target_date,
        "status": "ready_for_audit_review" if not errors else "blocked",
        "ready_for_audit_review": not errors,
        "allow_next_layer_continue": False,
        "next_layer": "L5",
        "valid": not errors,
        "errors": errors,
        "hard_rules": ["no-BJ", "DuckDB-only", "one-table-one-file", "explicit-qfq", "fail-closed"],
        "contract": {
            "agent_runtime_route": args.agent_model_route,
            "active_input_assets": [report["inputs"]["factor_asset"], report["inputs"]["label_asset"]],
            "l3_feature_input": factor_input_provenance,
            "user_authorized_partial_exception": bool(partial_exception),
            "partial_exception": partial_exception,
            "active_l3_feature_sha256": report["inputs"]["factor_asset_sha256"],
            "active_l3_label_sha256": report["inputs"]["label_asset_sha256"],
            "label_maturity_max_trade_date": report["inputs"]["factor_status"]["label_max_trade_date"],
            "authoritative_target_domain": {"rows": expected_rows, "stocks": expected_stocks},
            "all4_common_key_count": validation["all4_common_key_count"],
            "active_output_assets": output_assets,
            "manifest_bindings": manifest_bindings,
            "canonical_digest_contract": validation["digest_contract"],
            "postwrite_validation": str(validation_path),
            "postwrite_validation_sha256": sha256(validation_path),
            "formal_report": str(report_path),
            "formal_report_sha256": sha256(report_path),
            "l3_handoff": str(l3_handoff_path),
            "l3_handoff_sha256": sha256(l3_handoff_path),
            "manifest_archives": report.get("manifest_archives", {}),
            "boundaries": {
                "no_training": True,
                "no_tuning": True,
                "no_signal": True,
                "no_backtest": True,
                "no_new_research_manifest": True,
            },
            "residual_risk": [
                {
                    "severity": "P2",
                    "item": "label_maturity",
                    "detail": report["inputs"]["factor_status"]["label_max_trade_date"],
                }
            ],
        },
        "evidence": {
            "report_dir": str(report_dir),
            "l3_handoff_status": l3_handoff.get("status"),
            "business_assets_modified_by_handoff": False,
            "prediction_values_recomputed_by_handoff": False,
        },
    }
    output_path = report_dir / f"l4_to_l5_handoff_contract_{target_date}.json"
    output_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
