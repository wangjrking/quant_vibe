"""Strict 2022-2024-only 5D fold-local original-metric early-stop candidate."""
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
OUT = DATA_DIR / "reports" / "model_agent_5d_fold_local_top_return_earlystop_dev2022_2024_20260830" / "build_r1"
CONTRACT = OUT.parent / "training_contract.json"
BASE = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"
TAIL_DAYS = 126
TOP_K = 10


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    result, _ = base.evaluate(frame)
    return {key: float(result[key]) for key in ("rank_ic_mean", "top1_excess_mean", "top3_excess_mean", "top5_excess_mean", "top10_excess_mean")}


def top_return_metric(validation_dates: np.ndarray):
    """Original production metric semantics, with dates scoped to this fold's tail."""
    dates = np.asarray(validation_dates, dtype=str)
    unique_dates = np.unique(dates)
    def metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        kept: list[np.ndarray] = []
        for date in unique_dates:
            positions = np.flatnonzero(dates == date)
            count = min(TOP_K, len(positions))
            top = positions[np.argpartition(y_pred[positions], -count)[-count:]]
            kept.append(y_true[top])
        if not kept:
            raise RuntimeError("empty validation date groups")
        return -float(np.mean(np.concatenate(kept)))
    return metric


def fold_local_params(metadata: dict, validation_dates: np.ndarray) -> dict:
    saved = metadata["xgb_params"]
    params = base.model_params(metadata)
    params.update({
        "n_estimators": int(saved["n_estimators"]),
        "early_stopping_rounds": int(saved["early_stopping_rounds"]),
        "eval_metric": top_return_metric(validation_dates),
    })
    return params


def split_tail(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = train["trade_date"].drop_duplicates().sort_values().tolist()
    if len(dates) <= TAIL_DAYS:
        raise RuntimeError("insufficient available train days for frozen validation tail")
    validation_dates = set(dates[-TAIL_DAYS:])
    fit = train.loc[~train["trade_date"].isin(validation_dates)].reset_index(drop=True)
    validation = train.loc[train["trade_date"].isin(validation_dates)].reset_index(drop=True)
    if fit.empty or validation.empty or fit["trade_date"].max() >= validation["trade_date"].min():
        raise RuntimeError("invalid train-tail split")
    return fit, validation


def main() -> int:
    if OUT.exists(): raise RuntimeError(f"refusing overwrite: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (contract["candidate_id"] != "expanding_pit_5d_fold_local_top_return_earlystop_v1" or contract["single_change"]["kind"] != "fold_local_train_tail_top_return_early_stopping" or contract["selection_and_sealing"]["confirmation_window"] != "2025 sealed_for_5d_candidate_only" or contract["selection_and_sealing"]["validation_window"] != "2026 closed_for_5d_candidate_only"): raise RuntimeError("contract drift")
    dates = base.read_open_dates(); _, prepared = base.preflight(dates); info = prepared["5d"]
    folds = [fold for fold in info["folds"] if fold["test_end"] <= "20241231"]
    if [fold["fold_id"] for fold in folds] != ["fold2022", "fold2023", "fold2024"]: raise RuntimeError("wrong development folds")
    metadata = info["metadata"]
    if metadata.get("sample_weight_config") != {"mode": "daily_top_quantile", "top_pct": 0.005, "top_multiplier": 10.0}: raise RuntimeError("production sample-weight drift")
    baseline = pd.read_parquet(BASE / "5d_oof.parquet", filters=[("trade_date", "<=", "20241231")])
    if baseline.trade_date.max() > "20241231": raise RuntimeError("confirmation read")
    OUT.mkdir(parents=True)
    base.dump_json(OUT / "preflight.json", {"contract_sha256": base.sha256_file(CONTRACT), "features": info["resolved"], "folds": folds, "production_weight_config": metadata["sample_weight_config"], "frozen_validation": {"mode": "available_train_tail_days", "tail_days": TAIL_DAYS, "metric": "per-date Top10 negative mean return", "early_stopping_rounds": int(metadata["xgb_params"]["early_stopping_rounds"])}, "confirmation_2025_read": False, "validation_2026_closed": True, "production_unchanged": True, "runtime_agent_route": contract["runtime_agent_route"]})
    fd = duckdb.connect(str(base.FEATURE_DB), read_only=True); ld = duckdb.connect(str(base.LABEL_DB), read_only=True); rows = []; fitted = []
    try:
        for fold in folds:
            train = base.query_features(fd, info["resolved"], fold["train_start"], fold["train_end"]).merge(base.query_labels(ld, "executable_5d_open_return", fold["train_start"], fold["train_end"]), on=["trade_date", "stock_code"], how="inner", validate="one_to_one").dropna(subset=["target"]).reset_index(drop=True)
            test = base.query_features(fd, info["resolved"], fold["test_start"], fold["test_end"]).merge(base.query_labels(ld, "executable_5d_open_return", fold["test_start"], fold["mature_label_cutoff"]), on=["trade_date", "stock_code"], how="left", validate="one_to_one")
            fit, validation = split_tail(train)
            validation_dates = validation["trade_date"].to_numpy(dtype=str)
            model = xgb.XGBRegressor(**fold_local_params(metadata, validation_dates))
            model.fit(fit[info["resolved"]].astype("float32"), fit.target.astype("float32"), sample_weight=base.daily_weights(fit, metadata["sample_weight_config"]), eval_set=[(validation[info["resolved"]].astype("float32"), validation.target.astype("float32"))], verbose=False)
            matrix = test[info["resolved"]].astype("float32")
            first, second, third = (model.predict(matrix).astype("float64") for _ in range(3))
            if not (np.array_equal(first, second) and np.array_equal(first, third)): raise RuntimeError(f"{fold['fold_id']}: deterministic failure")
            scored = test[["trade_date", "stock_code", "target"]].copy(); scored["pred_prob"] = first; scored["fold_id"] = fold["fold_id"]; scored["label_mature_within_dev"] = scored.trade_date <= fold["mature_label_cutoff"]
            if scored.duplicated(["trade_date", "stock_code"]).any() or scored.stock_code.astype(str).str.endswith(".BJ").any() or scored.pred_prob.isna().any() or not np.isfinite(scored.pred_prob).all(): raise RuntimeError(f"{fold['fold_id']}: quality failure")
            rows.append(scored); fitted.append({"fold_id": fold["fold_id"], "fit_rows": len(fit), "validation_rows": len(validation), "fit_end": str(fit.trade_date.max()), "validation_start": str(validation.trade_date.min()), "validation_end": str(validation.trade_date.max()), "best_iteration": int(model.best_iteration), "best_score": float(model.best_score)})
    finally:
        fd.close(); ld.close()
    candidate = pd.concat(rows, ignore_index=True); candidate.to_parquet(OUT / "candidate_oof.parquet", index=False)
    be = baseline.loc[baseline.label_mature_within_dev, ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    ce = candidate.loc[candidate.label_mature_within_dev, ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    if not be[["trade_date", "stock_code"]].equals(ce[["trade_date", "stock_code"]]): raise RuntimeError("same key failure")
    comparisons = []
    for fid in ("fold2022", "fold2023", "fold2024"):
        before, after = metrics(be.query("fold_id == @fid")), metrics(ce.query("fold_id == @fid")); comparisons.append({"fold_id": fid, "baseline": before, "candidate": after, "delta": {k: after[k]-before[k] for k in before}})
    before, after = metrics(be), metrics(ce); delta = {k: after[k]-before[k] for k in before}
    gate = {"aggregate_rank_ic_not_weaker": delta["rank_ic_mean"] >= 0, "each_fold_rank_ic_not_weaker": all(x["delta"]["rank_ic_mean"] >= 0 for x in comparisons), "aggregate_top1_top3_top5_top10_not_weaker": all(delta[f"top{k}_excess_mean"] >= 0 for k in (1,3,5,10)), "each_fold_top10_not_weaker": all(x["delta"]["top10_excess_mean"] >= 0 for x in comparisons)}
    summary = {"candidate_id": contract["candidate_id"], "stage": "development_2022_2024_only", "same_key_rows": len(ce), "fitted_models": fitted, "aggregate_baseline": before, "aggregate_candidate": after, "aggregate_delta": delta, "fold_comparisons": comparisons, "acceptance_gate": gate, "decision": "development_pass_waiting_for_2025_confirmation" if all(gate.values()) else "reject_no_further_search", "confirmation_2025_read": False, "validation_2026_closed": True, "production_unchanged": True, "allow_next_layer_continue": False}
    base.dump_json(OUT / "evaluation_summary.json", summary); base.dump_json(OUT / "audit_handoff.json", {"status": summary["decision"], "candidate_oof_sha256": base.sha256_file(OUT / "candidate_oof.parquet"), "summary_sha256": base.sha256_file(OUT / "evaluation_summary.json"), "contract_sha256": base.sha256_file(CONTRACT), "confirmation_2025_read": False, "allow_next_layer_continue": False, "production_unchanged": True})
    return 0

if __name__ == "__main__": raise SystemExit(main())
