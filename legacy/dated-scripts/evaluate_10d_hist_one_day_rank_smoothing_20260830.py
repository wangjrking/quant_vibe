"""Frozen one-day rank smoothing of the strict 10D HistGradient OOF scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_expanding_pit_oof_baselines_20260829 import canonical_frame_hash, evaluate


SOURCE = Path("quant/data_file/reports/model_agent_10d_hist_gradient_structure_20260830/build_r1/baseline_candidate_same_key_oof.parquet")
OUT = Path("quant/data_file/reports/model_agent_10d_hist_one_day_rank_smoothing_20260830")


def metrics(frame: pd.DataFrame) -> dict[str, object]:
    value, _ = evaluate(frame[["trade_date", "stock_code", "target", "pred_prob"]])
    return value


def failures(base: dict[str, object], candidate: dict[str, object]) -> list[str]:
    result = []
    if candidate["rank_ic_mean"] < base["rank_ic_mean"]: result.append("rank_ic_not_weaker")
    for top in (1, 3, 5, 10):
        if candidate[f"top{top}_excess_mean"] < base[f"top{top}_excess_mean"]: result.append(f"top{top}_not_weaker")
    if candidate["top10_turnover_proxy"] > base["top10_turnover_proxy"]: result.append("top10_turnover_not_higher")
    return result


def main() -> int:
    if OUT.exists(): raise RuntimeError("blocked_fail_closed_output_exists")
    raw = pd.read_parquet(SOURCE).sort_values(["fold_id", "trade_date", "stock_code"]).reset_index(drop=True)
    if raw.duplicated(["fold_id", "trade_date", "stock_code"]).any() or raw.stock_code.astype(str).str.endswith(".BJ").any():
        raise RuntimeError("blocked_fail_closed_input_keys")
    raw["today_rank"] = raw.groupby(["fold_id", "trade_date"])["candidate_pred_prob"].rank(pct=True, method="average")
    days = raw[["fold_id", "trade_date"]].drop_duplicates().sort_values(["fold_id", "trade_date"])
    days["previous_trade_date"] = days.groupby("fold_id")["trade_date"].shift(1)
    work = raw.merge(days, on=["fold_id", "trade_date"], how="left", validate="many_to_one")
    prior = raw[["fold_id", "trade_date", "stock_code", "today_rank"]].rename(columns={"trade_date": "previous_trade_date", "today_rank": "previous_rank"})
    work = work.merge(prior, on=["fold_id", "previous_trade_date", "stock_code"], how="left", validate="many_to_one")
    work["smoothed_pred_prob"] = np.where(work.previous_rank.notna(), (work.today_rank + work.previous_rank) / 2.0, work.today_rank)
    if not np.isfinite(work.smoothed_pred_prob).all(): raise RuntimeError("blocked_fail_closed_nonfinite")
    replay = [canonical_frame_hash(work.rename(columns={"smoothed_pred_prob": "pred_prob"}), ["trade_date", "stock_code", "pred_prob"]) for _ in range(3)]
    folds = {}
    for fold, group in work.groupby("fold_id", sort=True):
        base = metrics(group.rename(columns={"baseline_pred_prob": "pred_prob"}))
        candidate = metrics(group.rename(columns={"smoothed_pred_prob": "pred_prob"}))
        folds[fold] = {"baseline_metrics": base, "candidate_metrics": candidate, "failed_gates": failures(base, candidate)}
    base_all = metrics(work.rename(columns={"baseline_pred_prob": "pred_prob"}))
    candidate_all = metrics(work.rename(columns={"smoothed_pred_prob": "pred_prob"}))
    all_failures = sorted(set(failures(base_all, candidate_all) + [x for v in folds.values() for x in v["failed_gates"]]))
    OUT.mkdir(parents=True)
    work.to_parquet(OUT / "baseline_candidate_same_key_oof.parquet", index=False)
    summary = {"candidate_id": "v261_10d_hist_one_day_rank_smoothing_v1", "formula": "0.5 * today_hist_rank + 0.5 * previous_trade_day_hist_rank; first day uses today rank", "development_window": ["20220101", "20241231"], "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"}, "same_key_rows": int(len(work)), "deterministic_replay_hashes": replay, "aggregate": {"baseline_metrics": base_all, "candidate_metrics": candidate_all, "failed_gates": failures(base_all, candidate_all)}, "folds": folds, "failed_gates": all_failures, "status": "completed_waiting_for_fixed_readonly_audit" if not all_failures else "reject_no_further_search", "production_unchanged": True, "allow_next_layer_continue": False}
    (OUT / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
