"""Evaluate one frozen 3D tie-break candidate on strict PIT/OOF scores only.

Production-specification 3D OOF scores have many exact same-day ties.  This
research-only candidate preserves every strict 3D ordering between distinct
scores and uses strict same-day 1D OOF score only within exact 3D ties.  No
weight, threshold, learned calibration, or parameter search is involved.
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
DEFAULT_OUTPUT = Path("quant/data_file/reports/model_agent_3d_1d_tiebreak_strict_pit_20260830")
CANDIDATE_ID = "v261_3d_exact_tie_1d_rank_tiebreak_v1"
DEVELOPMENT_START = "20220101"
DEVELOPMENT_END = "20241231"


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


def load_input() -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        root = BASELINE_ROOT.as_posix()
        frame = connection.execute(
            f"""
            SELECT b.trade_date, b.stock_code, b.fold_id, b.target,
                   b.pred_prob AS baseline_3d_score, a.pred_prob AS strict_1d_tiebreak_score
            FROM read_parquet('{root}/3d_oof.parquet') b
            INNER JOIN read_parquet('{root}/1d_oof.parquet') a USING(trade_date, stock_code)
            WHERE b.trade_date BETWEEN ? AND ?
              AND b.label_mature_within_dev
            ORDER BY b.trade_date, b.stock_code
            """,
            [DEVELOPMENT_START, DEVELOPMENT_END],
        ).fetchdf()
    finally:
        connection.close()
    numeric = ["target", "baseline_3d_score", "strict_1d_tiebreak_score"]
    if frame.empty or frame.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("blocked_fail_closed_same_key_input")
    if frame.stock_code.astype(str).str.endswith(".BJ").any():
        raise RuntimeError("blocked_fail_closed_bj_input")
    if not np.isfinite(frame[numeric].to_numpy(dtype="float64")).all():
        raise RuntimeError("blocked_fail_closed_nonfinite_input")
    return frame


def apply_tiebreak(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ordered = result.sort_values(
        ["trade_date", "baseline_3d_score", "strict_1d_tiebreak_score", "stock_code"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).copy()
    ordered["candidate_position"] = ordered.groupby("trade_date", sort=False).cumcount() + 1
    ordered["candidate_raw_score"] = -ordered["candidate_position"].astype("float64")
    return ordered.sort_values(["trade_date", "stock_code"], kind="mergesort").reset_index(drop=True)


def daily_spearman(frame: pd.DataFrame, score: str) -> tuple[float, int]:
    values: list[float] = []
    for _, group in frame.groupby("trade_date", sort=True):
        if group[score].nunique() > 1 and group.target.nunique() > 1:
            values.append(float(group[score].rank(method="average").corr(group.target.rank(method="average"))))
    if not values:
        raise RuntimeError("blocked_fail_closed_no_metric_dates")
    return float(np.mean(values)), len(values)


def top_mean(frame: pd.DataFrame, score: str, count: int) -> float:
    values: list[float] = []
    for _, group in frame.groupby("trade_date", sort=True):
        values.append(float(group.sort_values([score, "stock_code"], ascending=[False, True], kind="mergesort").head(count).target.mean()))
    return float(np.mean(values))


def turnover(frame: pd.DataFrame, score: str) -> float:
    picks = []
    for _, group in frame.groupby("trade_date", sort=True):
        picks.append(set(group.sort_values([score, "stock_code"], ascending=[False, True], kind="mergesort").head(10).stock_code.astype(str)))
    return float(np.mean([1.0 - len(before & after) / 10.0 for before, after in zip(picks, picks[1:])]))


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    baseline_ic, dates = daily_spearman(frame, "baseline_3d_score")
    candidate_ic, _ = daily_spearman(frame, "candidate_raw_score")
    result: dict[str, float | int] = {
        "rows": int(len(frame)),
        "trade_days": dates,
        "baseline_rank_ic": baseline_ic,
        "candidate_rank_ic": candidate_ic,
        "rank_ic_delta": candidate_ic - baseline_ic,
        "baseline_top10_turnover_proxy": turnover(frame, "baseline_3d_score"),
        "candidate_top10_turnover_proxy": turnover(frame, "candidate_raw_score"),
    }
    for count in (1, 3, 5, 10):
        result[f"baseline_top{count}_mean_target"] = top_mean(frame, "baseline_3d_score", count)
        result[f"candidate_top{count}_mean_target"] = top_mean(frame, "candidate_raw_score", count)
    return result


def gate(values: dict[str, float | int]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if values["candidate_rank_ic"] < values["baseline_rank_ic"]:
        failures.append("rank_ic_not_weaker")
    for count in (1, 3, 5, 10):
        if values[f"candidate_top{count}_mean_target"] < values[f"baseline_top{count}_mean_target"]:
            failures.append(f"top{count}_not_weaker")
    if values["candidate_top10_turnover_proxy"] > values["baseline_top10_turnover_proxy"]:
        failures.append("top10_turnover_not_higher")
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    output = Path(parser.parse_args().output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "candidate_id": CANDIDATE_ID,
        "candidate_count": 1,
        "status_before_run": "frozen_single_candidate",
        "development_window": [DEVELOPMENT_START, DEVELOPMENT_END],
        "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
        "formula": "sort by 3d score descending; within exact same 3d score sort by strict same-day 1d score descending then stock_code ascending",
        "preserves_primary_non_tie_order": True,
        "free_parameters": 0,
        "hard_gates": ["same_key_finite_no_bj", "aggregate_and_each_fold_rank_ic_not_weaker", "aggregate_and_each_fold_top1_3_5_10_not_weaker", "aggregate_and_each_fold_top10_turnover_not_higher"],
        "production_unchanged": True,
        "allow_next_layer_continue": False,
        "runtime_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
    }
    dump_json(output / "training_contract.json", contract)
    sources = {key: sha256_file(BASELINE_ROOT / f"{key}_oof.parquet") for key in ("1d", "3d")}
    try:
        scored = apply_tiebreak(load_input())
        aggregate = metrics(scored)
        per_fold = {fold: metrics(part.copy()) for fold, part in scored.groupby("fold_id", sort=True)}
        passed_aggregate, aggregate_failures = gate(aggregate)
        per_fold_gates = {fold: gate(value) for fold, value in per_fold.items()}
        passed = passed_aggregate and all(result[0] for result in per_fold_gates.values())
        failures = sorted(set(aggregate_failures + [item for _, (_, items) in per_fold_gates.items() for item in items]))
        scored.to_parquet(output / "baseline_candidate_same_key_oof.parquet", index=False)
        summary = {
            "candidate_id": CANDIDATE_ID,
            "status": "completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search",
            "development_window": [DEVELOPMENT_START, DEVELOPMENT_END],
            "read_windows": [DEVELOPMENT_START, DEVELOPMENT_END],
            "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
            "aggregate_metrics": aggregate,
            "fold_metrics": per_fold,
            "hard_gate_passed": passed,
            "failed_gates": failures,
            "same_key_rows": int(len(scored)),
            "same_key_duplicate_groups": int(scored.duplicated(["trade_date", "stock_code"]).sum()),
            "bj_rows": int(scored.stock_code.astype(str).str.endswith(".BJ").sum()),
            "null_or_nonfinite_rows": int((~np.isfinite(scored[["target", "baseline_3d_score", "strict_1d_tiebreak_score", "candidate_raw_score"]].to_numpy(dtype="float64")).all(axis=1)).sum()),
            "baseline_score_sha256": canonical_hash(scored, ["trade_date", "stock_code", "baseline_3d_score"]),
            "candidate_score_sha256": canonical_hash(scored, ["trade_date", "stock_code", "candidate_raw_score"]),
            "source_hashes": sources,
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
            "source_hashes": sources,
        })
        return 0
    except Exception as error:
        dump_json(output / "blocked_or_rejected.json", {"candidate_id": CANDIDATE_ID, "status": "blocked_fail_closed", "error": str(error), "production_unchanged": True, "allow_next_layer_continue": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
