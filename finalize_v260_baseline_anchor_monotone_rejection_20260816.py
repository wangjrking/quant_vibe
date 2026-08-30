"""Write a metadata-only terminal rejection record for the completed v260 Build."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(
    "quant/data_file/reports/"
    "model_agent_v260_baseline_anchor_monotone_excess_logit_10d_v1_20260816_r1"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    evaluation_path = ROOT / "evaluation_summary.json"
    manifest_path = ROOT / "research_candidate_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gates = evaluation["acceptance_gates"]
    failed = [name for name, passed in gates.items() if not passed]
    expected = {
        "baseline_rank_correlation_floor",
        "top10_overlap_floor",
        "turnover_proxy_not_up",
    }
    if set(failed) != expected or evaluation["deterministic_replay"]["passed"] is not True:
        raise SystemExit("unexpected rejection evidence; refusing metadata closure")
    closure = {
        "candidate_id": evaluation["candidate_id"],
        "terminal_status": "reject_no_further_search",
        "reason": "frozen_strategy_drift_gates_failed",
        "failed_frozen_gates": failed,
        "aggregate_failure_evidence": {
            "baseline_candidate_rank_correlation": evaluation["aggregate"]["baseline_candidate_rank_correlation"],
            "required_rank_correlation_floor": 0.70,
            "top10_overlap": evaluation["aggregate"]["top10_overlap"],
            "required_top10_overlap_floor": 0.30,
            "baseline_turnover_proxy": evaluation["aggregate"]["baseline_turnover_proxy"],
            "candidate_turnover_proxy": evaluation["aggregate"]["candidate_turnover_proxy"],
        },
        "deterministic_replay_3_of_3": True,
        "same_key_rows": evaluation["same_key"]["rows"],
        "next_actions_forbidden": ["strategy_ab", "validation", "production", "formal_publish", "further_search"],
        "production_unchanged": manifest["production_unchanged"],
        "allow_next_layer_continue": False,
        "evidence": {
            "evaluation_summary": {"path": str(evaluation_path), "sha256": sha256(evaluation_path)},
            "research_candidate_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        },
        "metadata_only_closure": True,
    }
    target = ROOT / "rejection_closure.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(closure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)
    print(json.dumps({"path": str(target), "sha256": sha256(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
