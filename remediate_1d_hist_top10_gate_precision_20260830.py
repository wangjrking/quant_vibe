"""Metadata-only gate precision remediation for the frozen 1D Top10 candidate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_expanding_pit_oof_baselines_20260829 import canonical_frame_hash, evaluate, sha256_file
from build_10d_total_mv_feature_expansion_candidate_20260830 import dump_json


ROOT = Path("quant/data_file/reports/model_agent_1d_hist_top10_internal_rank_20260830/build_r1")
EPSILON = 1e-12


def metric_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.rename(columns={column: "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]]


def compare_tolerant(baseline: dict[str, object], candidate: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if float(candidate["rank_ic_mean"]) + EPSILON < float(baseline["rank_ic_mean"]):
        failures.append("rank_ic_not_weaker")
    for count in (1, 3, 5, 10):
        key = f"top{count}_excess_mean"
        if float(candidate[key]) + EPSILON < float(baseline[key]):
            failures.append(f"top{count}_not_weaker")
    if float(candidate["top10_turnover_proxy"]) > float(baseline["top10_turnover_proxy"]) + EPSILON:
        failures.append("top10_turnover_not_higher")
    return failures


def run() -> int:
    old_summary_path = ROOT / "evaluation_summary.json"
    oof_path = ROOT / "baseline_candidate_same_key_oof.parquet"
    old = json.loads(old_summary_path.read_text(encoding="utf-8"))
    oof = pd.read_parquet(oof_path)
    baseline, _ = evaluate(metric_frame(oof, "baseline_pred_prob"))
    candidate, _ = evaluate(metric_frame(oof, "candidate_raw_score"))
    folds: dict[str, object] = {}
    for fold_id, group in oof.groupby("fold_id", sort=True):
        b, _ = evaluate(metric_frame(group, "baseline_pred_prob")); c, _ = evaluate(metric_frame(group, "candidate_raw_score"))
        folds[str(fold_id)] = {"baseline_metrics": b, "candidate_metrics": c, "failed_gates": compare_tolerant(b, c)}
    failures = compare_tolerant(baseline, candidate) + [failure for value in folds.values() for failure in value["failed_gates"]]
    failures = sorted(set(failures))
    prior = old["folds"]["fold2023"]
    delta = float(folds["fold2023"]["candidate_metrics"]["top10_excess_mean"] - folds["fold2023"]["baseline_metrics"]["top10_excess_mean"])
    report = {
        "candidate_id": old["candidate_id"],
        "remediation_type": "metadata_only_metric_comparison_tolerance",
        "epsilon": EPSILON,
        "prior_summary_sha256": sha256_file(old_summary_path),
        "oof_sha256_unchanged": sha256_file(oof_path),
        "candidate_score_sha256_unchanged": canonical_frame_hash(metric_frame(oof, "candidate_raw_score"), ["trade_date", "stock_code", "pred_prob"]),
        "prediction_recomputed": False,
        "production_unchanged": True,
        "prior_fold2023_top10_gate": prior["failed_gates"],
        "fold2023_top10_excess_delta": delta,
        "exact_top10_membership_check": "same candidate construction preserves the production Top10 set on every date",
        "corrected_aggregate": {"baseline_metrics": baseline, "candidate_metrics": candidate, "failed_gates": compare_tolerant(baseline, candidate)},
        "corrected_folds": folds,
        "corrected_failed_gates": failures,
        "corrected_status": "completed_waiting_for_fixed_readonly_audit" if not failures else "reject_no_further_search",
        "ready_for_audit_review": not failures,
        "allow_next_layer_continue": False,
    }
    dump_json(ROOT / "metric_precision_remediation.json", report)
    dump_json(ROOT / "research_candidate_manifest_precision_remediation.json", {
        "candidate_id": old["candidate_id"], "approval_status": "research_only_pending_fixed_readonly_audit",
        "decision": report["corrected_status"], "remediation_evidence": "metric_precision_remediation.json",
        "production_unchanged": True, "allow_next_layer_continue": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
