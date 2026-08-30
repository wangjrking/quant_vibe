"""Evaluate one zero-parameter strict-PIT 3D exact-score tie-break candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_ID = "v262_3d_hgb_exact_tie_break_v1"
CONTRACT = Path("quant/data_file/reports/model_agent_3d_hgb_tie_break_20260830/training_contract.json")
SOURCE = Path("quant/data_file/reports/model_agent_3d_hist_gradient_structure_20260830/build_r1/baseline_candidate_same_key_oof.parquet")
OUT = Path("quant/data_file/reports/model_agent_3d_hgb_tie_break_20260830/build_r1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values(["trade_date", "stock_code"], kind="mergesort")[columns].itertuples(index=False, name=None):
        digest.update(("|".join(format(value, ".17g") if isinstance(value, float) else str(value) for value in row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def evaluate(frame: pd.DataFrame) -> dict[str, float | int | None]:
    daily: list[dict[str, float]] = []
    sets: list[set[str]] = []
    for _, group in frame.groupby("trade_date", sort=True):
        ranked = group.sort_values(["pred_prob", "stock_code"], ascending=[False, True], kind="mergesort")
        mean = float(ranked.target.mean())
        row = {"rank_ic": float(ranked.pred_prob.rank(method="average").corr(ranked.target.rank(method="average")))}
        for count in (1, 3, 5, 10):
            row[f"top{count}"] = float(ranked.head(count).target.mean() - mean)
        daily.append(row); sets.append(set(ranked.head(10).stock_code))
    summary: dict[str, float | int | None] = {"trade_days": len(daily), "rank_ic_mean": float(np.mean([row["rank_ic"] for row in daily]))}
    for count in (1, 3, 5, 10):
        summary[f"top{count}_excess_mean"] = float(np.mean([row[f"top{count}"] for row in daily]))
    summary["top10_turnover_proxy"] = None if len(sets) < 2 else float(1.0 - np.mean([len(left & right) / 10.0 for left, right in zip(sets, sets[1:])]))
    return summary


def project(source: pd.DataFrame) -> pd.DataFrame:
    result: list[pd.DataFrame] = []
    for _, day in source.groupby("trade_date", sort=True):
        baseline = day.sort_values(["baseline_pred_prob", "stock_code"], ascending=[False, True], kind="mergesort")
        groups = [part.sort_values(["candidate_pred_prob", "stock_code"], ascending=[False, True], kind="mergesort") for _, part in baseline.groupby("baseline_pred_prob", sort=False)]
        final = pd.concat(groups, ignore_index=True)
        if not np.array_equal(final.groupby("baseline_pred_prob", sort=False).size().to_numpy(), baseline.groupby("baseline_pred_prob", sort=False).size().to_numpy()):
            raise RuntimeError("blocked_tie_partition")
        final["baseline_rank"] = final.stock_code.map({code: index + 1 for index, code in enumerate(baseline.stock_code)}).astype("int32")
        final["final_rank"] = np.arange(1, len(final) + 1, dtype="int32")
        final["candidate_raw_score"] = (len(final) - final.final_rank).astype("float64")
        result.append(final)
    return pd.concat(result, ignore_index=True).sort_values(["trade_date", "stock_code"], kind="mergesort").reset_index(drop=True)


def metric_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.rename(columns={column: "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]]


def failed(base: dict[str, float | int | None], candidate: dict[str, float | int | None]) -> list[str]:
    result: list[str] = []
    if candidate["rank_ic_mean"] < base["rank_ic_mean"]: result.append("rank_ic_not_weaker")
    for count in (1, 3, 5, 10):
        if candidate[f"top{count}_excess_mean"] < base[f"top{count}_excess_mean"]: result.append(f"top{count}_not_weaker")
    if candidate["top10_turnover_proxy"] > base["top10_turnover_proxy"]: result.append("top10_turnover_not_higher")
    return result


def main() -> int:
    if OUT.exists(): raise RuntimeError("blocked_existing_output")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["candidate_id"] != CANDIDATE_ID or contract["algorithm"]["free_parameters"] != 0: raise RuntimeError("blocked_contract")
    source = pd.read_parquet(SOURCE)
    required = {"trade_date", "stock_code", "target", "baseline_pred_prob", "candidate_pred_prob", "fold_id"}
    if not required.issubset(source.columns) or source.trade_date.astype(str).str[:4].isin(["2025", "2026"]).any(): raise RuntimeError("blocked_source_window")
    if source.duplicated(["trade_date", "stock_code"]).any() or source.stock_code.astype(str).str.endswith(".BJ").any() or not np.isfinite(source[["target", "baseline_pred_prob", "candidate_pred_prob"]].to_numpy(dtype="float64")).all(): raise RuntimeError("blocked_source_quality")
    replays = [project(source) for _ in range(3)]
    hashes = [frame_hash(item, ["trade_date", "stock_code", "candidate_raw_score"]) for item in replays]
    if len(set(hashes)) != 1: raise RuntimeError("blocked_determinism")
    oof = replays[0]
    aggregate_base, aggregate_candidate = evaluate(metric_frame(oof, "baseline_pred_prob")), evaluate(metric_frame(oof, "candidate_raw_score"))
    folds = {}
    for fold, group in oof.groupby("fold_id", sort=True):
        base, candidate = evaluate(metric_frame(group, "baseline_pred_prob")), evaluate(metric_frame(group, "candidate_raw_score"))
        folds[str(fold)] = {"rows": int(len(group)), "baseline_metrics": base, "candidate_metrics": candidate, "failed_gates": failed(base, candidate)}
    gates = sorted(set(failed(aggregate_base, aggregate_candidate) + [gate for fold in folds.values() for gate in fold["failed_gates"]]))
    changed = float((oof.baseline_rank != oof.final_rank).mean())
    if changed == 0: gates.append("nonzero_effective_change")
    OUT.mkdir(parents=True); oof.to_parquet(OUT / "baseline_candidate_same_key_oof.parquet", index=False)
    summary = {"candidate_id": CANDIDATE_ID, "status": "completed_waiting_for_fixed_readonly_audit" if not gates else "reject_no_further_search", "development_window": ["20220101", "20241231"], "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"}, "same_key_rows": int(len(oof)), "duplicate_key_groups": int(oof.duplicated(["trade_date", "stock_code"]).sum()), "bj_rows": 0, "null_or_nonfinite_rows": 0, "projection": {"exact_score_tie_break_only": True, "changed_rank_row_share": changed}, "aggregate": {"baseline_metrics": aggregate_base, "candidate_metrics": aggregate_candidate, "failed_gates": failed(aggregate_base, aggregate_candidate)}, "folds": folds, "failed_gates": sorted(set(gates)), "hard_gate_passed": not gates, "deterministic": {"passed": True, "replay_candidate_score_sha256": hashes}, "production_unchanged": True, "allow_next_layer_continue": False}
    (OUT / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "hash_inventory.json").write_text(json.dumps({"script_sha256": sha256(Path(__file__)), "contract_sha256": sha256(CONTRACT), "source_sha256": sha256(SOURCE), "oof_sha256": sha256(OUT / "baseline_candidate_same_key_oof.parquet"), "summary_sha256": sha256(OUT / "evaluation_summary.json")}, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
