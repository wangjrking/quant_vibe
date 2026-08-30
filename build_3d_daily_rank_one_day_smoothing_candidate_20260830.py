"""Evaluate a frozen one-day rank smoothing projection of strict 3D OOF.

This research-only projection starts from the strict PIT daily-rank 3D model
candidate.  On each fold, it averages a stock's current daily percentile rank
with its immediately preceding observed OOF-day rank; the first day uses its
current rank.  There are no weights, thresholds, refits, or additional data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_ID = "v261_3d_daily_rank_one_day_rank_smoothing_v1"
SOURCE_ROOT = Path("quant/data_file/reports/model_agent_3d_daily_percentile_rank_label_20260830")
DEFAULT_OUTPUT = Path("quant/data_file/reports/model_agent_3d_daily_rank_one_day_smoothing_20260830")
DEVELOPMENT_WINDOW = ["20220101", "20241231"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values(["trade_date", "stock_code"], kind="mergesort")[columns].itertuples(index=False, name=None):
        digest.update("|".join(format(value, ".17g") if isinstance(value, float) else str(value) for value in row).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_source() -> pd.DataFrame:
    frame = pd.read_parquet(SOURCE_ROOT / "baseline_candidate_same_key_oof.parquet")
    required = {"trade_date", "stock_code", "fold_id", "target", "baseline_pred_prob", "candidate_pred_prob"}
    if required.difference(frame.columns) or frame.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("blocked_fail_closed_source_schema_or_key")
    if frame.stock_code.astype(str).str.endswith(".BJ").any():
        raise RuntimeError("blocked_fail_closed_source_bj")
    if not np.isfinite(frame[["target", "baseline_pred_prob", "candidate_pred_prob"]].to_numpy(dtype="float64")).all():
        raise RuntimeError("blocked_fail_closed_source_finite")
    return frame


def project(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().sort_values(["fold_id", "trade_date", "stock_code"], kind="mergesort")
    result["current_rank_pct"] = result.groupby(["fold_id", "trade_date"], sort=False)["candidate_pred_prob"].rank(method="average", pct=True)
    result["previous_rank_pct"] = result.groupby(["fold_id", "stock_code"], sort=False)["current_rank_pct"].shift(1)
    result["candidate_raw_score"] = 0.5 * (result["current_rank_pct"] + result["previous_rank_pct"].fillna(result["current_rank_pct"]))
    if not np.isfinite(result["candidate_raw_score"]).all():
        raise RuntimeError("blocked_fail_closed_projection_nonfinite")
    return result


def daily_spearman(frame: pd.DataFrame, score: str) -> float:
    values: list[float] = []
    for _, group in frame.groupby("trade_date", sort=True):
        if group[score].nunique() > 1 and group.target.nunique() > 1:
            values.append(float(group[score].rank(method="average").corr(group.target.rank(method="average"))))
    if not values:
        raise RuntimeError("blocked_fail_closed_no_metric_dates")
    return float(np.mean(values))


def top_mean(frame: pd.DataFrame, score: str, count: int) -> float:
    values = []
    for _, group in frame.groupby("trade_date", sort=True):
        values.append(float(group.sort_values([score, "stock_code"], ascending=[False, True], kind="mergesort").head(count).target.mean()))
    return float(np.mean(values))


def turnover(frame: pd.DataFrame, score: str) -> float:
    picks = []
    for _, group in frame.groupby("trade_date", sort=True):
        picks.append(set(group.sort_values([score, "stock_code"], ascending=[False, True], kind="mergesort").head(10).stock_code.astype(str)))
    return float(np.mean([1.0 - len(left & right) / 10.0 for left, right in zip(picks, picks[1:])]))


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "rows": int(len(frame)),
        "baseline_rank_ic": daily_spearman(frame, "baseline_pred_prob"),
        "candidate_rank_ic": daily_spearman(frame, "candidate_raw_score"),
        "baseline_top10_turnover_proxy": turnover(frame, "baseline_pred_prob"),
        "candidate_top10_turnover_proxy": turnover(frame, "candidate_raw_score"),
    }
    result["rank_ic_delta"] = result["candidate_rank_ic"] - result["baseline_rank_ic"]
    for count in (1, 3, 5, 10):
        result[f"baseline_top{count}_mean_target"] = top_mean(frame, "baseline_pred_prob", count)
        result[f"candidate_top{count}_mean_target"] = top_mean(frame, "candidate_raw_score", count)
    return result


def gate(value: dict[str, float | int]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if value["candidate_rank_ic"] < value["baseline_rank_ic"]:
        failures.append("rank_ic_not_weaker")
    for count in (1, 3, 5, 10):
        if value[f"candidate_top{count}_mean_target"] < value[f"baseline_top{count}_mean_target"]:
            failures.append(f"top{count}_not_weaker")
    if value["candidate_top10_turnover_proxy"] > value["baseline_top10_turnover_proxy"]:
        failures.append("top10_turnover_not_higher")
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    output = Path(parser.parse_args().output_dir)
    output.mkdir(parents=True, exist_ok=False)
    contract = {
        "candidate_id": CANDIDATE_ID,
        "candidate_count": 1,
        "status_before_run": "frozen_single_candidate",
        "development_window": DEVELOPMENT_WINDOW,
        "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
        "formula": "0.5 * (current daily percentile rank + immediately prior observed OOF-day percentile rank); first day=current rank",
        "free_parameters": 0,
        "source": "strict PIT 3D daily-percentile-label candidate OOF only",
        "hard_gates": ["same_key_finite_no_bj", "aggregate_and_each_fold_rank_ic_not_weaker", "aggregate_and_each_fold_top1_3_5_10_not_weaker", "aggregate_and_each_fold_top10_turnover_not_higher"],
        "production_unchanged": True,
        "allow_next_layer_continue": False,
        "runtime_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
    }
    dump_json(output / "training_contract.json", contract)
    try:
        scored = project(load_source())
        aggregate = metrics(scored)
        per_fold = {fold: metrics(group.copy()) for fold, group in scored.groupby("fold_id", sort=True)}
        aggregate_gate = gate(aggregate)
        fold_gates = {fold: gate(value) for fold, value in per_fold.items()}
        passed = aggregate_gate[0] and all(value[0] for value in fold_gates.values())
        failures = sorted(set(aggregate_gate[1] + [item for _, (_, items) in fold_gates.items() for item in items]))
        scored.to_parquet(output / "baseline_candidate_same_key_oof.parquet", index=False)
        summary = {
            "candidate_id": CANDIDATE_ID,
            "status": "completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search",
            "development_window": DEVELOPMENT_WINDOW,
            "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
            "aggregate_metrics": aggregate,
            "fold_metrics": per_fold,
            "hard_gate_passed": passed,
            "failed_gates": failures,
            "same_key_rows": int(len(scored)),
            "same_key_duplicate_groups": int(scored.duplicated(["trade_date", "stock_code"]).sum()),
            "bj_rows": int(scored.stock_code.astype(str).str.endswith(".BJ").sum()),
            "null_or_nonfinite_rows": int((~np.isfinite(scored[["target", "baseline_pred_prob", "candidate_pred_prob", "candidate_raw_score"]].to_numpy(dtype="float64")).all(axis=1)).sum()),
            "baseline_score_sha256": canonical_hash(scored, ["trade_date", "stock_code", "baseline_pred_prob"]),
            "candidate_score_sha256": canonical_hash(scored, ["trade_date", "stock_code", "candidate_raw_score"]),
            "source_oof_sha256": sha256_file(SOURCE_ROOT / "baseline_candidate_same_key_oof.parquet"),
            "production_unchanged": True,
            "allow_next_layer_continue": False,
        }
        dump_json(output / "evaluation_summary.json", summary)
        dump_json(output / "research_candidate_manifest.json", {**contract, **summary})
        dump_json(output / "hash_inventory.json", {
            "script_sha256": sha256_file(Path(__file__)),
            "training_contract_sha256": sha256_file(output / "training_contract.json"),
            "evaluation_summary_sha256": sha256_file(output / "evaluation_summary.json"),
            "oof_sha256": sha256_file(output / "baseline_candidate_same_key_oof.parquet"),
            "source_oof_sha256": summary["source_oof_sha256"],
        })
    except Exception as error:
        dump_json(output / "blocked_or_rejected.json", {"candidate_id": CANDIDATE_ID, "status": "blocked_fail_closed", "error": str(error), "production_unchanged": True, "allow_next_layer_continue": False})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
