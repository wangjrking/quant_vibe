"""Evaluate one frozen strict-PIT 5D/1D same-day rank consensus candidate."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from build_expanding_pit_oof_baselines_20260829 import canonical_frame_hash, evaluate


ROOT = Path("quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3")
OUTPUT = Path("quant/data_file/reports/model_agent_5d_1d_equal_rank_consensus_20260830")
FEATURE_DB = "quant/data_file/production_assets/duckdb/l3_feature_current.duckdb"
DEVELOPMENT_START = "20220101"
DEVELOPMENT_END = "20241231"
CANDIDATE_ID = "v261_5d_1d_equal_daily_rank_consensus_v1"


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def compare(base: dict[str, object], candidate: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if candidate["rank_ic_mean"] < base["rank_ic_mean"]:
        failures.append("rank_ic_not_weaker")
    for top in (1, 3, 5, 10):
        if candidate[f"top{top}_excess_mean"] < base[f"top{top}_excess_mean"]:
            failures.append(f"top{top}_not_weaker")
    if candidate["top10_turnover_proxy"] > base["top10_turnover_proxy"]:
        failures.append("top10_turnover_not_higher")
    return failures


def load(primary_horizon: str, market_cap_adjust: bool) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        root = ROOT.as_posix()
        if market_cap_adjust:
            connection.execute(f"ATTACH '{FEATURE_DB}' AS l3 (READ_ONLY)")
            primary_source = f"""(
                SELECT p.*, feature.total_mv
                FROM read_parquet('{root}/{primary_horizon}_oof.parquet') AS p
                LEFT JOIN l3.prod_l3_production_factor_parts_20260625 AS feature
                  USING (trade_date, stock_code)
              )"""
        else:
            primary_source = f"read_parquet('{root}/{primary_horizon}_oof.parquet')"
        return connection.execute(
            f"""
            SELECT p.fold_id, p.trade_date, p.stock_code, p.target,
                   p.pred_prob AS baseline_pred_prob, one.pred_prob AS one_day_pred_prob
                   {", p.total_mv" if market_cap_adjust else ""}
            FROM {primary_source} AS p
            JOIN read_parquet('{root}/1d_oof.parquet') AS one
              USING (fold_id, trade_date, stock_code)
            WHERE p.label_mature_within_dev AND one.label_mature_within_dev
              AND p.trade_date BETWEEN ? AND ?
            ORDER BY p.fold_id, p.trade_date, p.stock_code
            """,
            [DEVELOPMENT_START, DEVELOPMENT_END],
        ).fetchdf()
    finally:
        connection.close()


def expected_key_count(primary_horizon: str) -> int:
    connection = duckdb.connect()
    try:
        root = ROOT.as_posix()
        return int(connection.execute(
            f"SELECT count(*) FROM read_parquet('{root}/{primary_horizon}_oof.parquet') WHERE label_mature_within_dev AND trade_date BETWEEN ? AND ?",
            [DEVELOPMENT_START, DEVELOPMENT_END],
        ).fetchone()[0])
    finally:
        connection.close()


def rank_consensus(frame: pd.DataFrame, market_cap_adjust: bool) -> pd.DataFrame:
    result = frame.copy()
    result["baseline_rank"] = result.groupby("trade_date")["baseline_pred_prob"].rank(pct=True, method="average")
    result["one_day_rank"] = result.groupby("trade_date")["one_day_pred_prob"].rank(pct=True, method="average")
    if market_cap_adjust:
        valid_market_cap = result["total_mv"].notna() & (result["total_mv"] > 0)
        result["market_cap_adjustment_applied"] = valid_market_cap
        result["market_cap_rank"] = result.loc[valid_market_cap].groupby("trade_date")["total_mv"].rank(pct=True, method="average")
        result["candidate_pred_prob"] = result["baseline_rank"]
        result.loc[valid_market_cap, "candidate_pred_prob"] = (
            result.loc[valid_market_cap, "baseline_rank"] - result.loc[valid_market_cap, "market_cap_rank"]
        )
    else:
        result["candidate_pred_prob"] = (result["baseline_rank"] + result["one_day_rank"]) / 2.0
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-horizon", default="5d")
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--candidate-id", default=CANDIDATE_ID)
    parser.add_argument("--market-cap-adjust", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError("blocked_fail_closed_output_exists")
    output.mkdir(parents=True)
    raw = load(args.primary_horizon, args.market_cap_adjust)
    if raw.empty or len(raw) != expected_key_count(args.primary_horizon) or raw.duplicated(["fold_id", "trade_date", "stock_code"]).any():
        raise RuntimeError("blocked_fail_closed_same_key")
    if raw.stock_code.astype(str).str.endswith(".BJ").any() or not np.isfinite(raw[["target", "baseline_pred_prob", "one_day_pred_prob"]].to_numpy()).all():
        raise RuntimeError("blocked_fail_closed_input_quality")
    candidate = rank_consensus(raw, args.market_cap_adjust)
    replay_hashes = []
    for _ in range(3):
        replay_hashes.append(canonical_frame_hash(rank_consensus(raw, args.market_cap_adjust), ["trade_date", "stock_code", "candidate_pred_prob"]))
    if len(set(replay_hashes)) != 1:
        raise RuntimeError("blocked_fail_closed_deterministic_replay")
    folds: dict[str, object] = {}
    for fold_id, group in candidate.groupby("fold_id", sort=True):
        base, _ = evaluate(group.rename(columns={"baseline_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
        score, _ = evaluate(group.rename(columns={"candidate_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
        folds[str(fold_id)] = {"baseline_metrics": base, "candidate_metrics": score, "failed_gates": compare(base, score)}
    base_all, _ = evaluate(candidate.rename(columns={"baseline_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
    candidate_all, _ = evaluate(candidate.rename(columns={"candidate_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
    aggregate_failures = compare(base_all, candidate_all)
    failures = sorted(set(aggregate_failures + [gate for value in folds.values() for gate in value["failed_gates"]]))
    status = "completed_waiting_for_fixed_readonly_audit" if not failures else "reject_no_further_search"
    candidate.to_parquet(output / "baseline_candidate_same_key_oof.parquet", index=False)
    formula = f"daily_percentile_rank({args.primary_horizon}_strict_pit_oof) - daily_percentile_rank(total_mv)" if args.market_cap_adjust else f"0.5 * daily_percentile_rank({args.primary_horizon}_strict_pit_oof) + 0.5 * daily_percentile_rank(1d_strict_pit_oof)"
    contract = {"candidate_id": args.candidate_id, "candidate_count": 1, "development_window": [DEVELOPMENT_START, DEVELOPMENT_END], "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"}, "formula": formula, "market_cap_missing_semantics": "identity_to_baseline_rank" if args.market_cap_adjust else None, "free_parameters": 0, "production_unchanged": True, "allow_next_layer_continue": False}
    dump(output / "training_contract.json", contract)
    summary = {**contract, "status": status, "same_key_rows": int(len(candidate)), "same_key_duplicate_groups": 0, "bj_rows": 0, "null_or_nonfinite_rows": 0, "deterministic_replay_hashes": replay_hashes, "aggregate": {"baseline_metrics": base_all, "candidate_metrics": candidate_all, "failed_gates": aggregate_failures}, "folds": folds, "failed_gates": failures, "baseline_score_sha256": canonical_frame_hash(candidate.rename(columns={"baseline_pred_prob": "pred_prob"}), ["trade_date", "stock_code", "pred_prob"]), "candidate_score_sha256": replay_hashes[0]}
    dump(output / "evaluation_summary.json", summary)
    dump(output / "hash_inventory.json", {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "summary_sha256": hashlib.sha256((output / "evaluation_summary.json").read_bytes()).hexdigest(), "oof_sha256": hashlib.sha256((output / "baseline_candidate_same_key_oof.parquet").read_bytes()).hexdigest()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
