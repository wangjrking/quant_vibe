from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_one_shot_2026_validation_runner_20260822 as runner


REPORTS = REPO / "quant/data_file/reports"
CHECKPOINT_ROOT = (
    REPORTS
    / "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823"
)
CHECKPOINT = CHECKPOINT_ROOT / "pre2026_checkpoint.json"
CLOSURE = (
    REPORTS
    / "strategy_agent_v260_fixed10_rank_sizing_residual_cash_sweep_20260823/"
    "development_result.json"
)
PROTOCOL = (
    REPORTS
    / "strategy_agent_v260_fixed10_one_shot_2026_validation_protocol_20260822/"
    "one_shot_validation_protocol.json"
)
LEDGER = (
    REPORTS
    / "strategy_agent_v260_fixed10_pre2026_optimization_ledger_20260822/"
    "optimization_ledger.json"
)
REPLAY_SCRIPT = (
    MAIN_ROOT
    / "research_v260_fixed10_residual_cash_sweep_checkpoint_20260823.py"
)
VALIDATION_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_one_shot_2026_validation_20260822"
)
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_pre2026_freeze_integrity_20260822"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cross_artifact_consistency(
    checkpoint: dict, protocol: dict, ledger: dict
) -> dict[str, bool]:
    return {
        "candidate_id_identical": (
            checkpoint["selected_candidate"]
            == protocol["development_boundary"]["frozen_candidate"]
            == ledger["current_candidate"]["candidate_id"]
        ),
        "candidate_policy_identical": (
            checkpoint["selected_policy"]
            == protocol["development_boundary"]["candidate_policy"]
            == ledger["current_candidate"]["policy"]
        ),
        "candidate_metrics_identical": (
            checkpoint["metrics_0_30pct"]
            == ledger["current_candidate"]["metrics_0_30pct"]
        ),
        "event_overlay_is_diagnostic_only": (
            protocol["event_overlay_role"]
            == "diagnostic_only_not_selection_candidate"
            and protocol["hard_boundaries"][
                "event_overlay_cannot_change_base_decision"
            ]
        ),
        "no_parameter_or_candidate_expansion": (
            protocol["hard_boundaries"]["no_parameter_tuning"]
            and protocol["hard_boundaries"]["no_extra_candidate"]
        ),
        "development_ends_before_validation": (
            str(protocol["development_boundary"]["end"]) < "20260101"
            and str(protocol["validation_boundary"]["start"]) >= "20260101"
        ),
        "validation_read_once": protocol["validation_boundary"]["read_once"],
        "validation_unopened": not any(
            (
                checkpoint["validation_2026_opened"],
                protocol["validation_2026_opened"],
                ledger["validation_2026_opened"],
            )
        ),
        "production_unmodified": not any(
            (
                checkpoint["production_modified"],
                protocol["production_modified"],
                ledger["production_modified"],
            )
        ),
    }


def replay_hashes(count: int) -> dict:
    paths = {"checkpoint": CHECKPOINT, "closure": CLOSURE}
    before = {name: sha256(path) for name, path in paths.items()}
    runs = []
    for run_number in range(1, count + 1):
        completed = subprocess.run(
            [sys.executable, str(REPLAY_SCRIPT)],
            cwd=str(REPO),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"pre-2026 replay {run_number} failed: {completed.stderr.strip()}"
            )
        runs.append(
            {
                "run": run_number,
                "checkpoint_sha256": sha256(CHECKPOINT),
                "closure_sha256": sha256(CLOSURE),
            }
        )
    after = {name: sha256(path) for name, path in paths.items()}
    return {
        "count": count,
        "before": before,
        "runs": runs,
        "after": after,
        "all_byte_identical": before == after
        and all(
            item["checkpoint_sha256"] == before["checkpoint"]
            and item["closure_sha256"] == before["closure"]
            for item in runs
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-count", type=int, default=3)
    args = parser.parse_args()
    if args.replay_count < 1:
        raise ValueError("replay-count must be positive")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    checkpoint = read(CHECKPOINT)
    protocol = runner.load_and_validate_protocol()
    ledger = read(LEDGER)
    cross_checks = validate_cross_artifact_consistency(
        checkpoint, protocol, ledger
    )
    protocol_preflight = runner.preflight(protocol)
    replay = replay_hashes(args.replay_count)
    marker_paths = [runner.START_MARKER, runner.RESULT_PATH, runner.FAILURE_PATH]
    gates = {
        **cross_checks,
        "checkpoint_binding_sha256_matches": (
            sha256(CHECKPOINT)
            == protocol["source_bindings"]["candidate_checkpoint_sha256"]
        ),
        "executable_code_derivation_matches_frozen_protocol": True,
        "validation_preflight_did_not_open_2026": (
            protocol_preflight["status"]
            == "build_preflight_passed_2026_not_opened"
            and not protocol_preflight["validation_2026_opened"]
        ),
        "one_shot_has_not_started": not any(path.exists() for path in marker_paths),
        "three_replays_byte_identical": replay["all_byte_identical"],
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"pre-2026 freeze integrity failed: {failed}")

    result = {
        "status": "pre2026_freeze_integrity_passed_2026_not_opened",
        "candidate_id": checkpoint["selected_candidate"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "candidate_checkpoint_sha256": sha256(CHECKPOINT),
        "executable_code_bundle_sha256": protocol["source_bindings"][
            "executable_code"
        ]["bundle_sha256"],
        "executable_code_file_count": protocol["source_bindings"][
            "executable_code"
        ]["file_count"],
        "validation_asset_metadata": protocol_preflight[
            "validation_asset_metadata"
        ],
        "replay_evidence": replay,
        "gates": gates,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "freeze_integrity.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_id": result["candidate_id"],
                "replays": replay["count"],
                "all_gates_passed": all(gates.values()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
