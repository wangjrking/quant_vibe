"""Evaluate one frozen, research-only cross-horizon 1D score candidate.

The candidate does not fit a new model.  It combines same-day strict PIT/OOF
scores from the four production-specification baselines using their daily
percentile ranks and a geometric mean.  This makes it a deliberately
low-freedom test of whether all-horizon agreement adds usable 1D ranking
information.  It never reads 2025 or later and never touches formal assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


BASELINE_ROOT = Path("quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3")
DEFAULT_OUTPUT = Path("quant/data_file/reports/model_agent_1d_cross_horizon_geomean_rank_consensus_20260830")
CANDIDATE_ID = "v261_1d_cross_horizon_geomean_rank_consensus_v1"
DEVELOPMENT_START = "20220101"
DEVELOPMENT_END = "20241231"
SCORE_COLUMNS = ("score_1d", "score_3d", "score_5d", "score_10d")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.sort_values(["trade_date", "stock_code"], kind="mergesort")
    digest = hashlib.sha256()
    for row in ordered[columns].itertuples(index=False, name=None):
        digest.update("|".join(str(value) if not isinstance(value, float) else format(value, ".17g") for value in row).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_same_key_oof() -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        root = BASELINE_ROOT.as_posix()
        result = connection.execute(
            f"""
            SELECT a.trade_date, a.stock_code, a.fold_id, a.target,
                   a.pred_prob AS score_1d,
                   b.pred_prob AS score_3d,
                   c.pred_prob AS score_5d,
                   d.pred_prob AS score_10d
            FROM read_parquet('{root}/1d_oof.parquet') a
            INNER JOIN read_parquet('{root}/3d_oof.parquet') b USING(trade_date, stock_code)
            INNER JOIN read_parquet('{root}/5d_oof.parquet') c USING(trade_date, stock_code)
            INNER JOIN read_parquet('{root}/10d_oof.parquet') d USING(trade_date, stock_code)
            WHERE a.trade_date BETWEEN ? AND ?
              AND a.label_mature_within_dev
            ORDER BY a.trade_date, a.stock_code
            """,
            [DEVELOPMENT_START, DEVELOPMENT_END],
        ).fetchdf()
    finally:
        connection.close()
    if result.empty or result.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("blocked_fail_closed_same_key_input")
    if result.stock_code.astype(str).str.endswith(".BJ").any():
        raise RuntimeError("blocked_fail_closed_bj_input")
    if not np.isfinite(result[["target", *SCORE_COLUMNS]].to_numpy(dtype="float64")).all():
        raise RuntimeError("blocked_fail_closed_nonfinite_input")
    return result


def add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source in SCORE_COLUMNS:
        result[f"rank_{source}"] = result.groupby("trade_date", sort=False)[source].rank(method="average", pct=True)
    rank_columns = [f"rank_{source}" for source in SCORE_COLUMNS]
    result["candidate_raw_score"] = np.exp(np.log(result[rank_columns].clip(lower=1e-12)).mean(axis=1))
    if not np.isfinite(result["candidate_raw_score"]).all():
        raise RuntimeError("blocked_fail_closed_nonfinite_candidate")
    return result


def daily_spearman(frame: pd.DataFrame, score: str) -> tuple[float, int]:
    values: list[float] = []
    for _, group in frame.groupby("trade_date", sort=True):
        if group[score].nunique() > 1 and group.target.nunique() > 1:
            values.append(float(group[score].rank(method="average").corr(group.target.rank(method="average"))))
    if not values:
        raise RuntimeError("blocked_fail_closed_no_spearman_dates")
    return float(np.mean(values)), len(values)


def top_mean(frame: pd.DataFrame, score: str, count: int) -> float:
    values = []
    for _, group in frame.groupby("trade_date", sort=True):
        ranked = group.sort_values([score, "stock_code"], ascending=[False, True], kind="mergesort")
        values.append(float(ranked.head(count).target.mean()))
    return float(np.mean(values))


def top10_turnover(frame: pd.DataFrame, score: str) -> float:
    picks = []
    for date, group in frame.groupby("trade_date", sort=True):
        chosen = set(group.sort_values([score, "stock_code"], ascending=[False, True], kind="mergesort").head(10).stock_code.astype(str))
        picks.append((str(date), chosen))
    if len(picks) < 2:
        raise RuntimeError("blocked_fail_closed_no_turnover_dates")
    return float(np.mean([1.0 - len(now & prior) / 10.0 for (_, prior), (_, now) in zip(picks, picks[1:])]))


def evaluate(frame: pd.DataFrame) -> dict[str, float | int]:
    baseline_ic, days = daily_spearman(frame, "score_1d")
    candidate_ic, _ = daily_spearman(frame, "candidate_raw_score")
    return {
        "rows": int(len(frame)),
        "trade_days": int(days),
        "baseline_rank_ic": baseline_ic,
        "candidate_rank_ic": candidate_ic,
        "rank_ic_delta": candidate_ic - baseline_ic,
        **{
            f"baseline_top{count}_mean_target": top_mean(frame, "score_1d", count)
            for count in (1, 3, 5, 10)
        },
        **{
            f"candidate_top{count}_mean_target": top_mean(frame, "candidate_raw_score", count)
            for count in (1, 3, 5, 10)
        },
        "baseline_top10_turnover_proxy": top10_turnover(frame, "score_1d"),
        "candidate_top10_turnover_proxy": top10_turnover(frame, "candidate_raw_score"),
    }


def gate(metrics: dict[str, float | int]) -> tuple[bool, list[str]]:
    failed: list[str] = []
    if float(metrics["candidate_rank_ic"]) < float(metrics["baseline_rank_ic"]):
        failed.append("rank_ic_not_weaker")
    for count in (1, 3, 5, 10):
        if float(metrics[f"candidate_top{count}_mean_target"]) < float(metrics[f"baseline_top{count}_mean_target"]):
            failed.append(f"top{count}_not_weaker")
    if float(metrics["candidate_top10_turnover_proxy"]) > float(metrics["baseline_top10_turnover_proxy"]):
        failed.append("top10_turnover_not_higher")
    return not failed, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "candidate_id": CANDIDATE_ID,
        "candidate_count": 1,
        "status_before_run": "frozen_single_candidate",
        "development_window": [DEVELOPMENT_START, DEVELOPMENT_END],
        "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
        "formula": "geometric_mean(daily_percentile_rank(score_1d), score_3d, score_5d, score_10d)",
        "free_parameters": 0,
        "baseline": "strict_pit_oof_1d_production_specification",
        "hard_gates": ["same_key_finite_no_bj", "aggregate_and_each_fold_rank_ic_not_weaker", "aggregate_and_each_fold_top1_3_5_10_not_weaker", "aggregate_and_each_fold_top10_turnover_not_higher"],
        "production_unchanged": True,
        "allow_next_layer_continue": False,
        "runtime_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
    }
    dump_json(output / "training_contract.json", contract)
    source_hashes = {f"{horizon}_oof_sha256": sha256_file(BASELINE_ROOT / f"{horizon}_oof.parquet") for horizon in ("1d", "3d", "5d", "10d")}
    try:
        scored = add_scores(load_same_key_oof())
        fold_metrics = {fold: evaluate(group.copy()) for fold, group in scored.groupby("fold_id", sort=True)}
        aggregate_metrics = evaluate(scored)
        aggregate_pass, aggregate_failed = gate(aggregate_metrics)
        fold_gates = {fold: gate(metrics) for fold, metrics in fold_metrics.items()}
        passed = aggregate_pass and all(result[0] for result in fold_gates.values())
        failed = sorted(set(aggregate_failed + [item for _, (_, values) in fold_gates.items() for item in values]))
        scored.to_parquet(output / "baseline_candidate_same_key_oof.parquet", index=False)
        summary = {
            "candidate_id": CANDIDATE_ID,
            "status": "completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search",
            "development_window": [DEVELOPMENT_START, DEVELOPMENT_END],
            "read_windows": [DEVELOPMENT_START, DEVELOPMENT_END],
            "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
            "aggregate_metrics": aggregate_metrics,
            "fold_metrics": fold_metrics,
            "hard_gate_passed": passed,
            "failed_gates": failed,
            "same_key_rows": int(len(scored)),
            "same_key_duplicate_groups": int(scored.duplicated(["trade_date", "stock_code"]).sum()),
            "bj_rows": int(scored.stock_code.astype(str).str.endswith(".BJ").sum()),
            "null_or_nonfinite_rows": int((~np.isfinite(scored[["target", "score_1d", "score_3d", "score_5d", "score_10d", "candidate_raw_score"]].to_numpy(dtype="float64")).all(axis=1)).sum()),
            "baseline_score_sha256": canonical_hash(scored, ["trade_date", "stock_code", "score_1d"]),
            "candidate_score_sha256": canonical_hash(scored, ["trade_date", "stock_code", "candidate_raw_score"]),
            "source_hashes": source_hashes,
            "production_unchanged": True,
            "allow_next_layer_continue": False,
        }
        dump_json(output / "evaluation_summary.json", summary)
        dump_json(output / "research_candidate_manifest.json", {**contract, **summary})
        dump_json(output / "hash_inventory.json", {
            "script_sha256": sha256_file(Path(__file__)),
            "contract_sha256": sha256_file(output / "training_contract.json"),
            "summary_sha256": sha256_file(output / "evaluation_summary.json"),
            "oof_sha256": sha256_file(output / "baseline_candidate_same_key_oof.parquet"),
            **source_hashes,
        })
        return 0
    except Exception as error:
        dump_json(output / "blocked_or_rejected.json", {"candidate_id": CANDIDATE_ID, "status": "blocked_fail_closed", "error": str(error), "production_unchanged": True, "allow_next_layer_continue": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
