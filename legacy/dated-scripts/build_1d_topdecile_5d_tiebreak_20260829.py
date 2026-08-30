"""Evaluate one frozen strict-OOF 1D/5D deterministic rank tiebreak candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_expanding_pit_oof_baselines_20260829 as base


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
OUTPUT_DIR = DATA_DIR / "reports" / "model_agent_1d_topdecile_5d_tiebreak_20260829" / "build_r1"
CONTRACT_PATH = OUTPUT_DIR.parent / "training_contract.json"
BASELINE_ROOT = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"


def metric_subset(frame: pd.DataFrame) -> dict[str, float]:
    metrics, _ = base.evaluate(frame)
    return {key: float(metrics[key]) for key in ("rank_ic_mean", "top1_excess_mean", "top3_excess_mean", "top5_excess_mean", "top10_excess_mean")}


def tiebreak(frame: pd.DataFrame, top_band_pct: float) -> np.ndarray:
    """Return a deterministic ordinal score without using labels."""
    scored = np.empty(len(frame), dtype=np.float64)
    for _, group in frame.groupby("trade_date", sort=False):
        primary = group.sort_values(["score_1d", "stock_code"], ascending=[False, True], kind="mergesort")
        top_count = max(10, int(np.ceil(len(primary) * top_band_pct)))
        top = primary.head(top_count).sort_values(["score_5d", "score_1d", "stock_code"], ascending=[False, False, True], kind="mergesort")
        ordered = pd.concat([top, primary.iloc[top_count:]], ignore_index=False)
        if set(ordered.head(10)["stock_code"]) - set(primary.head(10)["stock_code"]):
            raise RuntimeError("Top10 membership changed")
        # Larger score means earlier deterministic rank.  Index alignment is
        # explicit because source groups retain their original row labels.
        scored[ordered.index.to_numpy()] = np.arange(len(ordered), 0, -1, dtype=np.float64)
    return scored


def main() -> int:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {OUTPUT_DIR}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    change = contract["single_change"]
    if contract["candidate_id"] != "expanding_pit_1d_topdecile_5d_tiebreak_v1" or not contract["development_window"]["validation_2026_closed"]:
        raise RuntimeError("invalid frozen contract")
    if change["top_band_pct"] != 0.1 or change["top10_membership"] != "preserve exact 1D daily Top10 membership":
        raise RuntimeError("contract semantics drifted")
    one = pd.read_parquet(BASELINE_ROOT / "1d_oof.parquet")
    five = pd.read_parquet(BASELINE_ROOT / "5d_oof.parquet", columns=["trade_date", "stock_code", "pred_prob"])
    if one["trade_date"].max() > base.DEVELOPMENT_END or five["trade_date"].max() > base.DEVELOPMENT_END:
        raise RuntimeError("forbidden validation date read")
    frame = one.merge(five.rename(columns={"pred_prob": "score_5d"}), on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    if len(frame) != len(one):
        raise RuntimeError("strict same-key input closure failed")
    frame = frame.rename(columns={"pred_prob": "score_1d"})
    frame["pred_prob"] = tiebreak(frame, change["top_band_pct"])
    # The function is deterministic by construction; make its replay evidence explicit.
    if not np.array_equal(frame["pred_prob"].to_numpy(), tiebreak(frame, change["top_band_pct"])):
        raise RuntimeError("deterministic replay failed")
    if (frame.duplicated(["trade_date", "stock_code"]).any() or frame["stock_code"].astype(str).str.endswith(".BJ").any()
            or frame["pred_prob"].isna().any() or not np.isfinite(frame["pred_prob"]).all()):
        raise RuntimeError("candidate quality gate failed")
    OUTPUT_DIR.mkdir(parents=True)
    candidate = frame[["trade_date", "stock_code", "target", "pred_prob", "fold_id", "label_mature_within_dev"]].copy()
    candidate.to_parquet(OUTPUT_DIR / "candidate_oof.parquet", index=False)
    base_eval = one.loc[one["label_mature_within_dev"], ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    candidate_eval = candidate.loc[candidate["label_mature_within_dev"], ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    if not base_eval[["trade_date", "stock_code"]].equals(candidate_eval[["trade_date", "stock_code"]]):
        raise RuntimeError("evaluated same-key closure failed")
    folds = []
    for fold_id in sorted(candidate_eval["fold_id"].unique()):
        left, right = metric_subset(base_eval.query("fold_id == @fold_id")), metric_subset(candidate_eval.query("fold_id == @fold_id"))
        folds.append({"fold_id": fold_id, "baseline": left, "candidate": right, "delta": {key: right[key] - left[key] for key in left}})
    left, right = metric_subset(base_eval), metric_subset(candidate_eval)
    delta = {key: right[key] - left[key] for key in left}
    gate = {
        "aggregate_rank_ic_not_weaker": delta["rank_ic_mean"] >= 0,
        "each_fold_rank_ic_not_weaker": all(item["delta"]["rank_ic_mean"] >= 0 for item in folds),
        "aggregate_top1_top3_top5_top10_not_weaker": all(delta[f"top{x}_excess_mean"] >= 0 for x in (1, 3, 5, 10)),
        "each_fold_top10_not_weaker": all(item["delta"]["top10_excess_mean"] >= 0 for item in folds),
    }
    summary = {"candidate_id": contract["candidate_id"], "approval_status": "research_only_not_for_l5", "same_key_rows": int(len(candidate_eval)), "fold_comparisons": folds, "aggregate_baseline": left, "aggregate_candidate": right, "aggregate_delta": delta, "acceptance_gate": gate, "decision": "completed_ready_for_audit" if all(gate.values()) else "reject_no_further_search", "ready_for_audit_review": True, "allow_next_layer_continue": False, "production_unchanged": True, "validation_2026_closed": True}
    base.dump_json(OUTPUT_DIR / "preflight.json", {"contract_sha256": base.sha256_file(CONTRACT_PATH), "baseline_1d_sha256": base.sha256_file(BASELINE_ROOT / "1d_oof.parquet"), "baseline_5d_sha256": base.sha256_file(BASELINE_ROOT / "5d_oof.parquet"), "development_closed_after": base.DEVELOPMENT_END, "validation_2026_closed": True, "production_unchanged": True, "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"}})
    base.dump_json(OUTPUT_DIR / "evaluation_summary.json", summary)
    base.dump_json(OUTPUT_DIR / "audit_handoff.json", {"status": summary["decision"], "candidate_oof_sha256": base.sha256_file(OUTPUT_DIR / "candidate_oof.parquet"), "summary_sha256": base.sha256_file(OUTPUT_DIR / "evaluation_summary.json"), "contract_sha256": base.sha256_file(CONTRACT_PATH), "ready_for_audit_review": True, "allow_next_layer_continue": False, "production_unchanged": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
