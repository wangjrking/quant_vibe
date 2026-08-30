"""Strict pre-2025 development build for one 3D feature-governance candidate."""

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
OUTPUT_DIR = DATA_DIR / "reports" / "model_agent_3d_drop_market_levels_dev2022_2024_20260830" / "build_r1"
CONTRACT_PATH = OUTPUT_DIR.parent / "training_contract.json"
BASELINE_ROOT = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"
MARKET_LEVELS = {"index_2000_open", "index_2000_high", "index_2000_low", "index_2000_close"}


def metric_subset(frame: pd.DataFrame) -> dict[str, float]:
    metrics, _ = base.evaluate(frame)
    return {key: float(metrics[key]) for key in ("rank_ic_mean", "top1_excess_mean", "top3_excess_mean", "top5_excess_mean", "top10_excess_mean")}


def main() -> int:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"refusing to overwrite candidate evidence: {OUTPUT_DIR}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["candidate_id"] != "expanding_pit_3d_drop_market_levels_v1" or contract["confirmation_window"] != "2025 sealed" or not contract["validation_2026_closed"]:
        raise RuntimeError("frozen contract boundary violation")
    open_dates = base.read_open_dates()
    _, prepared = base.preflight(open_dates)
    info = prepared["3d"]
    removed = [name for name in info["resolved"] if name in MARKET_LEVELS]
    if set(removed) != set(contract["single_change"]["removed_features"]) or len(removed) != len(contract["single_change"]["removed_features"]):
        raise RuntimeError("resolved feature removal drift")
    features = [name for name in info["resolved"] if name not in MARKET_LEVELS]
    folds = [fold for fold in info["folds"] if fold["test_end"] <= "20241231"]
    if [fold["fold_id"] for fold in folds] != ["fold2022", "fold2023", "fold2024"]:
        raise RuntimeError("design folds are not exactly 2022-2024")
    baseline = pd.read_parquet(BASELINE_ROOT / "3d_oof.parquet", filters=[("trade_date", "<=", "20241231")])
    if baseline["trade_date"].max() > "20241231":
        raise RuntimeError("confirmation data read")
    OUTPUT_DIR.mkdir(parents=True)
    base.dump_json(OUTPUT_DIR / "preflight.json", {"contract_sha256": base.sha256_file(CONTRACT_PATH), "baseline_source": str(BASELINE_ROOT / "3d_oof.parquet"), "production_metadata_sha256": base.sha256_file(base.HORIZONS[1].metadata_path), "kept_features": features, "removed_features": removed, "folds": folds, "confirmation_2025_read": False, "validation_2026_closed": True, "production_unchanged": True, "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"}})
    feature_db = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_db = duckdb.connect(str(base.LABEL_DB), read_only=True)
    frames = []
    try:
        for fold in folds:
            train = base.query_features(feature_db, features, fold["train_start"], fold["train_end"]).merge(base.query_labels(label_db, "executable_3d_open_return", fold["train_start"], fold["train_end"]), on=["trade_date", "stock_code"], how="inner", validate="one_to_one").dropna(subset=["target"]).reset_index(drop=True)
            test = base.query_features(feature_db, features, fold["test_start"], fold["test_end"]).merge(base.query_labels(label_db, "executable_3d_open_return", fold["test_start"], fold["mature_label_cutoff"]), on=["trade_date", "stock_code"], how="left", validate="one_to_one")
            model = xgb.XGBRegressor(**base.model_params(info["metadata"]))
            weights = base.daily_weights(train, info["metadata"].get("sample_weight_config"))
            model.fit(train[features].astype("float32"), train["target"].astype("float32"), sample_weight=weights, verbose=False)
            first = model.predict(test[features].astype("float32")).astype("float64")
            second = model.predict(test[features].astype("float32")).astype("float64")
            third = model.predict(test[features].astype("float32")).astype("float64")
            if not (np.array_equal(first, second) and np.array_equal(first, third)):
                raise RuntimeError(f"{fold['fold_id']}: deterministic replay failed")
            scored = test[["trade_date", "stock_code", "target"]].copy()
            scored["pred_prob"], scored["fold_id"] = first, fold["fold_id"]
            scored["label_mature_within_dev"] = scored["trade_date"] <= fold["mature_label_cutoff"]
            if scored.duplicated(["trade_date", "stock_code"]).any() or scored["stock_code"].astype(str).str.endswith(".BJ").any() or scored["pred_prob"].isna().any() or not np.isfinite(scored["pred_prob"]).all():
                raise RuntimeError(f"{fold['fold_id']}: quality gate failed")
            frames.append(scored)
    finally:
        feature_db.close(); label_db.close()
    candidate = pd.concat(frames, ignore_index=True)
    candidate.to_parquet(OUTPUT_DIR / "candidate_oof.parquet", index=False)
    base_eval = baseline.loc[baseline["label_mature_within_dev"], ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    candidate_eval = candidate.loc[candidate["label_mature_within_dev"], ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    if not base_eval[["trade_date", "stock_code"]].equals(candidate_eval[["trade_date", "stock_code"]]):
        raise RuntimeError("same-key closure failed")
    comparisons = []
    for fold_id in ["fold2022", "fold2023", "fold2024"]:
        left, right = metric_subset(base_eval.query("fold_id == @fold_id")), metric_subset(candidate_eval.query("fold_id == @fold_id"))
        comparisons.append({"fold_id": fold_id, "baseline": left, "candidate": right, "delta": {key: right[key] - left[key] for key in left}})
    left, right = metric_subset(base_eval), metric_subset(candidate_eval)
    delta = {key: right[key] - left[key] for key in left}
    gate = {"aggregate_rank_ic_not_weaker": delta["rank_ic_mean"] >= 0, "each_fold_rank_ic_not_weaker": all(item["delta"]["rank_ic_mean"] >= 0 for item in comparisons), "aggregate_top1_top3_top5_top10_not_weaker": all(delta[f"top{x}_excess_mean"] >= 0 for x in (1, 3, 5, 10)), "each_fold_top10_not_weaker": all(item["delta"]["top10_excess_mean"] >= 0 for item in comparisons)}
    summary = {"candidate_id": contract["candidate_id"], "stage": "development_2022_2024_only", "approval_status": "research_only_not_for_l5", "same_key_rows": int(len(candidate_eval)), "fold_comparisons": comparisons, "aggregate_baseline": left, "aggregate_candidate": right, "aggregate_delta": delta, "acceptance_gate": gate, "decision": "development_pass_waiting_for_2025_confirmation" if all(gate.values()) else "reject_no_further_search", "confirmation_2025_read": False, "validation_2026_closed": True, "production_unchanged": True, "allow_next_layer_continue": False}
    base.dump_json(OUTPUT_DIR / "evaluation_summary.json", summary)
    base.dump_json(OUTPUT_DIR / "audit_handoff.json", {"status": summary["decision"], "candidate_oof_sha256": base.sha256_file(OUTPUT_DIR / "candidate_oof.parquet"), "summary_sha256": base.sha256_file(OUTPUT_DIR / "evaluation_summary.json"), "contract_sha256": base.sha256_file(CONTRACT_PATH), "confirmation_2025_read": False, "allow_next_layer_continue": False, "production_unchanged": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
