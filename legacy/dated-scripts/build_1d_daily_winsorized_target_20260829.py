"""Single frozen, research-only 1D daily-winsorized target candidate."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

import build_expanding_pit_oof_baselines_20260829 as base

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
OUTPUT_DIR = DATA_DIR / "reports" / "model_agent_1d_daily_winsorized_target_20260829" / "build_r1"
CONTRACT_PATH = OUTPUT_DIR.parent / "training_contract.json"
BASELINE_ROOT = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"


def winsorized_target(frame: pd.DataFrame) -> pd.Series:
    def clip_day(group: pd.Series) -> pd.Series:
        lower, upper = group.quantile(0.01), group.quantile(0.99)
        return group.clip(lower=lower, upper=upper)
    return frame.groupby("trade_date", sort=False)["target"].transform(clip_day)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    result, _ = base.evaluate(frame)
    return {key: float(result[key]) for key in ("rank_ic_mean", "top1_excess_mean", "top3_excess_mean", "top5_excess_mean", "top10_excess_mean")}


def main() -> int:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"refusing to overwrite candidate evidence: {OUTPUT_DIR}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("candidate_id") != "expanding_pit_1d_daily_winsorized_raw_return_v1" or not contract.get("validation_2026_closed"):
        raise RuntimeError("invalid frozen contract")
    baseline = pd.read_parquet(BASELINE_ROOT / "1d_oof.parquet")
    if baseline["trade_date"].max() > base.DEVELOPMENT_END:
        raise RuntimeError("baseline boundary violation")
    dates = base.read_open_dates()
    _, prepared = base.preflight(dates)
    info = prepared["1d"]
    metadata = info["metadata"]
    OUTPUT_DIR.mkdir(parents=True)
    base.dump_json(OUTPUT_DIR / "preflight.json", {
        "contract_sha256": base.sha256_file(CONTRACT_PATH),
        "baseline_sha256": base.sha256_file(BASELINE_ROOT / "1d_oof.parquet"),
        "production_metadata_sha256": base.sha256_file(base.HORIZONS[0].metadata_path),
        "folds": info["folds"], "resolved_features": info["resolved"],
        "development_closed_after": base.DEVELOPMENT_END, "validation_2026_closed": True,
        "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"}, "production_unchanged": True,
    })
    feature_db = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_db = duckdb.connect(str(base.LABEL_DB), read_only=True)
    frames = []
    try:
        for fold in info["folds"]:
            train = base.query_features(feature_db, info["resolved"], fold["train_start"], fold["train_end"]).merge(
                base.query_labels(label_db, "executable_1d_open_return", fold["train_start"], fold["train_end"]), on=["trade_date", "stock_code"], how="inner", validate="one_to_one").dropna(subset=["target"]).copy()
            test = base.query_features(feature_db, info["resolved"], fold["test_start"], fold["test_end"]).merge(
                base.query_labels(label_db, "executable_1d_open_return", fold["test_start"], fold["mature_label_cutoff"]), on=["trade_date", "stock_code"], how="left", validate="one_to_one")
            model = xgb.XGBRegressor(**base.model_params(metadata))
            target = winsorized_target(train)
            model.fit(train[info["resolved"]].astype("float32"), target.astype("float32"), verbose=False)
            first = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
            second = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
            third = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
            if not (np.array_equal(first, second) and np.array_equal(first, third)):
                raise RuntimeError(f"{fold['fold_id']}: deterministic replay failed")
            scored = test[["trade_date", "stock_code", "target"]].copy()
            scored["pred_prob"], scored["fold_id"] = first, fold["fold_id"]
            scored["label_mature_within_dev"] = scored["trade_date"] <= fold["mature_label_cutoff"]
            if scored.duplicated(["trade_date", "stock_code"]).any() or scored["stock_code"].str.endswith(".BJ").any() or scored["pred_prob"].isna().any() or not np.isfinite(scored["pred_prob"]).all():
                raise RuntimeError(f"{fold['fold_id']}: quality gate failed")
            frames.append(scored)
    finally:
        feature_db.close(); label_db.close()
    candidate = pd.concat(frames, ignore_index=True)
    base_eval = baseline.loc[baseline["label_mature_within_dev"], ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    candidate_eval = candidate.loc[candidate["label_mature_within_dev"], ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    if not base_eval[["trade_date", "stock_code"]].equals(candidate_eval[["trade_date", "stock_code"]]):
        raise RuntimeError("same-key closure failed")
    comparisons = []
    for fold_id in sorted(candidate_eval["fold_id"].unique()):
        left, right = metrics(base_eval.query("fold_id == @fold_id")), metrics(candidate_eval.query("fold_id == @fold_id"))
        comparisons.append({"fold_id": fold_id, "baseline": left, "candidate": right, "delta": {key: right[key]-left[key] for key in left}})
    left, right = metrics(base_eval), metrics(candidate_eval)
    delta = {key: right[key]-left[key] for key in left}
    gate = {"aggregate_rank_ic_not_weaker": delta["rank_ic_mean"] >= 0, "each_fold_rank_ic_not_weaker": all(x["delta"]["rank_ic_mean"] >= 0 for x in comparisons), "aggregate_top1_top3_top5_top10_not_weaker": all(delta[f"top{x}_excess_mean"] >= 0 for x in (1,3,5,10)), "each_fold_top10_not_weaker": all(x["delta"]["top10_excess_mean"] >= 0 for x in comparisons)}
    candidate.to_parquet(OUTPUT_DIR / "candidate_oof.parquet", index=False)
    summary = {"candidate_id": contract["candidate_id"], "approval_status": "research_only_not_for_l5", "same_key_rows": int(len(candidate_eval)), "fold_comparisons": comparisons, "aggregate_baseline": left, "aggregate_candidate": right, "aggregate_delta": delta, "acceptance_gate": gate, "decision": "completed_ready_for_audit" if all(gate.values()) else "reject_no_further_search", "ready_for_audit_review": True, "allow_next_layer_continue": False, "production_unchanged": True, "validation_2026_closed": True}
    base.dump_json(OUTPUT_DIR / "evaluation_summary.json", summary)
    base.dump_json(OUTPUT_DIR / "audit_handoff.json", {"status": summary["decision"], "candidate_oof_sha256": base.sha256_file(OUTPUT_DIR / "candidate_oof.parquet"), "summary_sha256": base.sha256_file(OUTPUT_DIR / "evaluation_summary.json"), "contract_sha256": base.sha256_file(CONTRACT_PATH), "ready_for_audit_review": True, "allow_next_layer_continue": False, "production_unchanged": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
