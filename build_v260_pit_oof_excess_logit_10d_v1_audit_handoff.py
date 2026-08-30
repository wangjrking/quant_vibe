"""Build metadata-only audit handoff for the completed v260 research candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_ROOT = Path(
    "quant/data_file/reports/model_agent_v260_pit_oof_excess_logit_10d_v1_20260816_r3"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.report_root.resolve()

    evaluation_path = root / "evaluation_summary.json"
    manifest_path = root / "research_candidate_manifest.json"
    preflight_path = root / "preflight.json"
    inventory_path = root / "hash_inventory.json"
    oof_path = root / "baseline_candidate_same_key_oof.parquet"
    required = [evaluation_path, manifest_path, preflight_path, inventory_path, oof_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required evidence: {missing}")

    evaluation = load_json(evaluation_path)
    manifest = load_json(manifest_path)
    preflight = load_json(preflight_path)
    inventory = load_json(inventory_path)
    gates = evaluation.get("model_layer_acceptance_gates", {})
    required_gates = [
        "same_key_coverage",
        "pit_oof_integrity",
        "binary_discrimination",
        "return_rank_sanity",
        "score_finiteness",
        "deterministic_replay_3_of_3",
    ]
    errors: list[str] = []
    if evaluation.get("decision") != "strategy_development_ab_only_pending_audit":
        errors.append("unexpected evaluation decision")
    if evaluation.get("ready_for_audit_review") is not True:
        errors.append("evaluation is not audit-ready")
    if evaluation.get("allow_next_layer_continue") is not False:
        errors.append("evaluation must keep next layer closed")
    if evaluation.get("sealed_2025_2026_not_read") is not True:
        errors.append("sealed 2025/2026 boundary is not proven")
    if manifest.get("approval_status") != "research_only_not_for_l5":
        errors.append("manifest is not research-only")
    if manifest.get("production_unchanged") is not True:
        errors.append("production unchanged flag is missing")
    if any(gates.get(gate) is not True for gate in required_gates):
        errors.append("one or more model-layer gates failed")
    if errors:
        raise SystemExit("; ".join(errors))

    handoff = {
        "candidate_id": evaluation["candidate_id"],
        "contract_sha256": evaluation["contract_sha256"],
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_for_l5",
        "decision": evaluation["decision"],
        "review_scope": [
            "contract v2 adherence",
            "official calendar-only session arithmetic",
            "strict 2022-2024 PIT/OOF folds with 2021 lag-state only",
            "sealed 2025/2026 non-read boundary",
            "same-key OOF closure and score finiteness",
            "model-layer acceptance gates and deterministic 3/3 replay",
        ],
        "candidate_output": {
            "path": str(oof_path),
            "sha256": sha256(oof_path),
            "same_key": evaluation["same_key"],
        },
        "aggregate": evaluation["aggregate"],
        "fold_metrics": evaluation["fold_metrics"],
        "model_layer_acceptance_gates": gates,
        "deterministic_replay": evaluation["deterministic_replay"],
        "boundary_assertions": {
            "research_only": True,
            "production_unchanged": True,
            "strategy_ab_not_run": True,
            "validation_2025_2026_not_read": True,
            "allow_next_layer_continue": False,
        },
        "evidence": {
            "evaluation_summary": {"path": str(evaluation_path), "sha256": sha256(evaluation_path)},
            "research_candidate_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "preflight": {"path": str(preflight_path), "sha256": sha256(preflight_path)},
            "hash_inventory": {"path": str(inventory_path), "sha256": sha256(inventory_path)},
            "official_calendar": preflight["calendar"],
        },
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    output = root / "audit_handoff.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"audit_handoff": str(output), "sha256": sha256(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
