"""Strict 2022-2024-only daily-top-decile 1D classifier research candidate."""
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
OUT = DATA_DIR / "reports" / "model_agent_1d_daily_top_decile_classifier_dev2022_2024_20260830" / "build_r1"
CONTRACT = OUT.parent / "training_contract.json"
BASE = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    result, _ = base.evaluate(frame)
    return {key: float(result[key]) for key in ("rank_ic_mean", "top1_excess_mean", "top3_excess_mean", "top5_excess_mean", "top10_excess_mean")}


def daily_top_decile_target(frame: pd.DataFrame) -> pd.Series:
    percentile = frame.groupby("trade_date", sort=False).target.rank(method="average", pct=True)
    result = (percentile >= 0.90).astype("int8")
    if result.isna().any() or result.min() != 0 or result.max() != 1:
        raise RuntimeError("invalid daily top-decile target")
    if not (0.08 <= float(result.mean()) <= 0.12):
        raise RuntimeError("unexpected daily top-decile class balance")
    return result


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing overwrite: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (
        contract["candidate_id"] != "expanding_pit_1d_daily_top_decile_classifier_v1"
        or contract["single_change"]["kind"] != "daily_top_decile_binary_target"
        or contract["selection_and_sealing"]["confirmation_window"] != "2025 sealed_for_1d_candidate_only"
        or contract["selection_and_sealing"]["validation_window"] != "2026 closed_for_1d_candidate_only"
    ):
        raise RuntimeError("contract drift")

    dates = base.read_open_dates()
    _, prepared = base.preflight(dates)
    info = prepared["1d"]
    folds = [fold for fold in info["folds"] if fold["test_end"] <= "20241231"]
    if [fold["fold_id"] for fold in folds] != ["fold2022", "fold2023", "fold2024"]:
        raise RuntimeError("wrong development folds")
    if info["metadata"].get("sample_weight_config") is not None:
        raise RuntimeError("unexpected production 1D sample weighting")
    baseline = pd.read_parquet(BASE / "1d_oof.parquet", filters=[("trade_date", "<=", "20241231")])
    if baseline.trade_date.max() > "20241231":
        raise RuntimeError("confirmation read")

    OUT.mkdir(parents=True)
    base.dump_json(OUT / "preflight.json", {
        "contract_sha256": base.sha256_file(CONTRACT),
        "features": info["resolved"],
        "folds": folds,
        "production_weight_config": None,
        "candidate_target_transform": "per-day mature executable_1d_open_return rank(method=average,pct=true) >= 0.90",
        "candidate_estimator": "XGBClassifier(binary:logistic) with unchanged production tree and regularization params",
        "confirmation_2025_read": False,
        "validation_2026_closed": True,
        "production_unchanged": True,
        "runtime_agent_route": contract["runtime_agent_route"],
    })

    feature_db = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_db = duckdb.connect(str(base.LABEL_DB), read_only=True)
    rows: list[pd.DataFrame] = []
    try:
        for fold in folds:
            train = base.query_features(feature_db, info["resolved"], fold["train_start"], fold["train_end"]).merge(
                base.query_labels(label_db, "executable_1d_open_return", fold["train_start"], fold["train_end"]),
                on=["trade_date", "stock_code"], how="inner", validate="one_to_one",
            ).dropna(subset=["target"]).reset_index(drop=True)
            test = base.query_features(feature_db, info["resolved"], fold["test_start"], fold["test_end"]).merge(
                base.query_labels(label_db, "executable_1d_open_return", fold["test_start"], fold["mature_label_cutoff"]),
                on=["trade_date", "stock_code"], how="left", validate="one_to_one",
            )
            params = base.model_params(info["metadata"])
            params["objective"] = "binary:logistic"
            model = xgb.XGBClassifier(**params)
            target = daily_top_decile_target(train)
            model.fit(train[info["resolved"]].astype("float32"), target, verbose=False)
            first = model.predict_proba(test[info["resolved"]].astype("float32"))[:, 1].astype("float64")
            second = model.predict_proba(test[info["resolved"]].astype("float32"))[:, 1].astype("float64")
            third = model.predict_proba(test[info["resolved"]].astype("float32"))[:, 1].astype("float64")
            if not (np.array_equal(first, second) and np.array_equal(first, third)):
                raise RuntimeError(f"{fold['fold_id']}: deterministic failure")
            scored = test[["trade_date", "stock_code", "target"]].copy()
            scored["pred_prob"] = first
            scored["fold_id"] = fold["fold_id"]
            scored["label_mature_within_dev"] = scored.trade_date <= fold["mature_label_cutoff"]
            if (scored.duplicated(["trade_date", "stock_code"]).any() or scored.stock_code.astype(str).str.endswith(".BJ").any() or scored.pred_prob.isna().any() or not np.isfinite(scored.pred_prob).all()):
                raise RuntimeError(f"{fold['fold_id']}: quality failure")
            rows.append(scored)
    finally:
        feature_db.close()
        label_db.close()

    candidate = pd.concat(rows, ignore_index=True)
    candidate.to_parquet(OUT / "candidate_oof.parquet", index=False)
    baseline_eval = baseline.loc[baseline.label_mature_within_dev, ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    candidate_eval = candidate.loc[candidate.label_mature_within_dev, ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    if not baseline_eval[["trade_date", "stock_code"]].equals(candidate_eval[["trade_date", "stock_code"]]):
        raise RuntimeError("same key failure")

    comparisons = []
    for fold_id in ("fold2022", "fold2023", "fold2024"):
        before, after = metrics(baseline_eval.query("fold_id == @fold_id")), metrics(candidate_eval.query("fold_id == @fold_id"))
        comparisons.append({"fold_id": fold_id, "baseline": before, "candidate": after, "delta": {key: after[key] - before[key] for key in before}})
    before, after = metrics(baseline_eval), metrics(candidate_eval)
    delta = {key: after[key] - before[key] for key in before}
    gate = {
        "aggregate_rank_ic_not_weaker": delta["rank_ic_mean"] >= 0,
        "each_fold_rank_ic_not_weaker": all(item["delta"]["rank_ic_mean"] >= 0 for item in comparisons),
        "aggregate_top1_top3_top5_top10_not_weaker": all(delta[f"top{count}_excess_mean"] >= 0 for count in (1, 3, 5, 10)),
        "each_fold_top10_not_weaker": all(item["delta"]["top10_excess_mean"] >= 0 for item in comparisons),
    }
    summary = {
        "candidate_id": contract["candidate_id"], "stage": "development_2022_2024_only", "same_key_rows": len(candidate_eval),
        "fold_comparisons": comparisons, "aggregate_baseline": before, "aggregate_candidate": after,
        "aggregate_delta": delta, "acceptance_gate": gate,
        "decision": "development_pass_waiting_for_2025_confirmation" if all(gate.values()) else "reject_no_further_search",
        "confirmation_2025_read": False, "validation_2026_closed": True,
        "production_unchanged": True, "allow_next_layer_continue": False,
    }
    base.dump_json(OUT / "evaluation_summary.json", summary)
    base.dump_json(OUT / "audit_handoff.json", {
        "status": summary["decision"], "candidate_oof_sha256": base.sha256_file(OUT / "candidate_oof.parquet"),
        "summary_sha256": base.sha256_file(OUT / "evaluation_summary.json"), "contract_sha256": base.sha256_file(CONTRACT),
        "confirmation_2025_read": False, "allow_next_layer_continue": False, "production_unchanged": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
