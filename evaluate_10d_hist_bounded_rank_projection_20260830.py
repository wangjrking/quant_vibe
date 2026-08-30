"""Evaluate the single frozen bounded-rank projection of the 10D HGB OOF.

This is a deterministic post-processing research build.  It never trains a
model, reads only the already materialized 2022-2024 same-key OOF, and keeps
the production score as the initial daily order.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_expanding_pit_oof_baselines_20260829 import canonical_frame_hash, evaluate, sha256_file
from build_10d_total_mv_feature_expansion_candidate_20260830 import compare_metrics, dump_json


CANDIDATE_ID = "v261_10d_hist_bounded_rank_projection_v1"
CONTRACT = Path("quant/data_file/reports/model_agent_10d_hist_bounded_rank_projection_20260830/training_contract.json")
HGB_OOF = Path("quant/data_file/reports/model_agent_10d_hist_gradient_structure_20260830/build_r1/baseline_candidate_same_key_oof.parquet")
DEFAULT_OUTPUT = Path("quant/data_file/reports/model_agent_10d_hist_bounded_rank_projection_20260830/build_r1")
PHASES = 6


def project_one_day(group: pd.DataFrame) -> pd.DataFrame:
    """Keep the production order, then make exactly six odd/even HGB swaps."""
    base = group.sort_values(
        ["baseline_pred_prob", "stock_code"], ascending=[False, True], kind="mergesort"
    ).copy()
    desired = group.sort_values(
        ["candidate_pred_prob", "stock_code"], ascending=[False, True], kind="mergesort"
    )["stock_code"].tolist()
    desired_rank = {code: index for index, code in enumerate(desired)}
    work = base["stock_code"].tolist()
    for phase in range(PHASES):
        for position in range(phase % 2, len(work) - 1, 2):
            left, right = work[position], work[position + 1]
            if desired_rank[right] < desired_rank[left]:
                work[position], work[position + 1] = right, left
    final_rank = {code: index + 1 for index, code in enumerate(work)}
    base["candidate_raw_score"] = base["stock_code"].map(final_rank).rsub(len(base)).astype("float64")
    base["baseline_rank"] = np.arange(1, len(base) + 1, dtype="int32")
    base["final_rank"] = base["stock_code"].map(final_rank).astype("int32")
    if (base["final_rank"].sub(base["baseline_rank"]).abs() > PHASES).any():
        raise RuntimeError("blocked_rank_projection_displacement")
    return base


def project(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [project_one_day(group) for _, group in frame.groupby("trade_date", sort=True)],
        ignore_index=True,
    ).sort_values(["trade_date", "stock_code"], kind="mergesort").reset_index(drop=True)


def replay_hash(frame: pd.DataFrame) -> str:
    return canonical_frame_hash(frame, ["trade_date", "stock_code", "candidate_raw_score"])


def run(output: Path) -> int:
    if output.exists():
        if any(output.iterdir()):
            raise RuntimeError("blocked_existing_output")
    else:
        output.mkdir(parents=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["candidate_id"] != CANDIDATE_ID or contract["algorithm"]["passes"] != PHASES:
        raise RuntimeError("blocked_contract_mismatch")
    source = pd.read_parquet(HGB_OOF)
    needed = {"trade_date", "stock_code", "target", "baseline_pred_prob", "candidate_pred_prob", "fold_id"}
    if not needed.issubset(source.columns):
        raise RuntimeError("blocked_hgb_oof_schema")
    if source["trade_date"].astype(str).str[:4].isin(["2025", "2026"]).any():
        raise RuntimeError("blocked_sealed_window_read")
    if source.duplicated(["trade_date", "stock_code"]).any() or source.stock_code.astype(str).str.endswith(".BJ").any():
        raise RuntimeError("blocked_source_key_or_bj")
    numeric = source[["target", "baseline_pred_prob", "candidate_pred_prob"]].to_numpy(dtype="float64")
    if not np.isfinite(numeric).all():
        raise RuntimeError("blocked_source_nonfinite")
    dump_json(output / "preflight.json", {
        "candidate_id": CANDIDATE_ID,
        "contract_sha256": sha256_file(CONTRACT),
        "hgb_oof_sha256": sha256_file(HGB_OOF),
        "development_window": ["20220101", "20241231"],
        "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
        "algorithm": "production_order_then_six_fixed_odd_even_hgb_preference_swaps",
        "production_unchanged": True,
    })
    replays = [project(source) for _ in range(3)]
    hashes = [replay_hash(frame) for frame in replays]
    if len(set(hashes)) != 1:
        raise RuntimeError("blocked_deterministic_replay")
    oof = replays[0]
    base_frame = oof.rename(columns={"baseline_pred_prob": "pred_prob"})[
        ["trade_date", "stock_code", "target", "pred_prob"]
    ]
    candidate_frame = oof.rename(columns={"candidate_raw_score": "pred_prob"})[
        ["trade_date", "stock_code", "target", "pred_prob"]
    ]
    base_aggregate, _ = evaluate(base_frame)
    candidate_aggregate, _ = evaluate(candidate_frame)
    fold_results: dict[str, object] = {}
    for fold_id, group in oof.groupby("fold_id", sort=True):
        baseline_metrics, _ = evaluate(group.rename(columns={"baseline_pred_prob": "pred_prob"})[
            ["trade_date", "stock_code", "target", "pred_prob"]
        ])
        candidate_metrics, _ = evaluate(group.rename(columns={"candidate_raw_score": "pred_prob"})[
            ["trade_date", "stock_code", "target", "pred_prob"]
        ])
        fold_results[str(fold_id)] = {
            "rows": int(len(group)),
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "failed_gates": compare_metrics(baseline_metrics, candidate_metrics),
        }
    aggregate_failures = compare_metrics(base_aggregate, candidate_aggregate)
    failures = sorted(set(aggregate_failures + [
        failure for item in fold_results.values() for failure in item["failed_gates"]
    ]))
    changed_share = float((oof["baseline_rank"] != oof["final_rank"]).mean())
    if changed_share == 0:
        failures.append("nonzero_effective_change")
    failures = sorted(set(failures))
    passed = not failures
    oof.to_parquet(output / "baseline_candidate_same_key_oof.parquet", index=False)
    summary = {
        "candidate_id": CANDIDATE_ID,
        "status": "completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search",
        "same_key_rows": int(len(oof)),
        "same_key_duplicate_groups": int(oof.duplicated(["trade_date", "stock_code"]).sum()),
        "bj_rows": int(oof.stock_code.astype(str).str.endswith(".BJ").sum()),
        "null_or_nonfinite_rows": int((~np.isfinite(oof[["target", "baseline_pred_prob", "candidate_raw_score"]].to_numpy(dtype="float64")).all(axis=1)).sum()),
        "projection": {"odd_even_phases": PHASES, "max_displacement": int((oof.final_rank - oof.baseline_rank).abs().max()), "changed_rank_row_share": changed_share},
        "aggregate": {"baseline_metrics": base_aggregate, "candidate_metrics": candidate_aggregate, "failed_gates": aggregate_failures},
        "folds": fold_results,
        "hard_gate_passed": passed,
        "failed_gates": failures,
        "deterministic": {"passed": True, "replay_candidate_score_sha256": hashes},
        "baseline_score_sha256": canonical_frame_hash(base_frame, ["trade_date", "stock_code", "pred_prob"]),
        "candidate_score_sha256": canonical_frame_hash(candidate_frame, ["trade_date", "stock_code", "pred_prob"]),
        "production_unchanged": True,
        "allow_next_layer_continue": False,
    }
    dump_json(output / "evaluation_summary.json", summary)
    dump_json(output / "research_candidate_manifest.json", {
        "candidate_id": CANDIDATE_ID,
        "approval_status": "research_only_not_for_l5",
        "decision": summary["status"],
        "production_unchanged": True,
        "allow_next_layer_continue": False,
    })
    dump_json(output / "hash_inventory.json", {
        "script_sha256": sha256_file(Path(__file__)),
        "contract_sha256": sha256_file(CONTRACT),
        "source_hgb_oof_sha256": sha256_file(HGB_OOF),
        "oof_sha256": sha256_file(output / "baseline_candidate_same_key_oof.parquet"),
        "summary_sha256": sha256_file(output / "evaluation_summary.json"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(run(DEFAULT_OUTPUT))
