"""Strict 2022-2024-only stationary-price 3D research candidate."""
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
OUT = DATA_DIR / "reports" / "model_agent_3d_stationary_price_ratio_dev2022_2024_20260830" / "build_r2"
CONTRACT = OUT.parent / "training_contract.json"
BASE = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"
INDEX_COLS = ["index_2000_open", "index_2000_high", "index_2000_low", "index_2000_close"]
STOCK_COLS = ["open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"]
NON_PRICE_COLS = ["amount", "turnover_rate_f", "index_2000_amount", "turnover_rate", "total_mv", "vol"]
CANDIDATE_COLS = [f"{column}_to_prev_index_close" for column in INDEX_COLS] + [f"{column}_to_pre_close" for column in STOCK_COLS[:-1]] + NON_PRICE_COLS


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    result, _ = base.evaluate(frame)
    return {key: float(result[key]) for key in ("rank_ic_mean", "top1_excess_mean", "top3_excess_mean", "top5_excess_mean", "top10_excess_mean")}


def prior_feature_date(connection: duckdb.DuckDBPyConnection, date: str) -> str | None:
    row = connection.execute(f"SELECT max(trade_date) FROM {base.quote(base.FEATURE_TABLE)} WHERE trade_date < ?", [date]).fetchone()
    return None if row is None or row[0] is None else str(row[0])


def transformed_features(connection: duckdb.DuckDBPyConnection, start: str, end: str) -> pd.DataFrame:
    prior = prior_feature_date(connection, start)
    query_start = prior or start
    raw_cols = INDEX_COLS + STOCK_COLS + NON_PRICE_COLS
    raw = base.query_features(connection, raw_cols, query_start, end)
    daily = raw.groupby("trade_date", sort=True)[INDEX_COLS].first().reset_index()
    for column in INDEX_COLS:
        if raw.groupby("trade_date", sort=True)[column].nunique(dropna=True).gt(1).any():
            raise RuntimeError(f"{column}: multiple non-null market values within a trade date")
    daily["prior_index_close"] = daily["index_2000_close"].shift(1)
    raw = raw.merge(daily[["trade_date", "prior_index_close"]], on="trade_date", how="left", validate="many_to_one")
    for column in INDEX_COLS:
        raw[f"{column}_to_prev_index_close"] = raw[column] / raw["prior_index_close"] - 1.0
    for column in STOCK_COLS[:-1]:
        raw[f"{column}_to_pre_close"] = raw[column] / raw["pre_close_qfq"] - 1.0
    result = raw.loc[raw.trade_date >= start, ["trade_date", "stock_code", *CANDIDATE_COLS]].reset_index(drop=True)
    values = result[CANDIDATE_COLS].to_numpy(dtype="float64", copy=False)
    if np.isinf(values).any():
        raise RuntimeError("relative-price transform produced infinity")
    return result


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing overwrite: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (
        contract["candidate_id"] != "expanding_pit_3d_stationary_price_ratio_v1"
        or contract["confirmation_window"] != "2025 sealed_for_3d_candidate_only"
        or not contract["validation_2026_closed_for_3d_candidate_only"]
    ):
        raise RuntimeError("contract drift")
    dates = base.read_open_dates()
    _, prepared = base.preflight(dates)
    info = prepared["3d"]
    folds = [fold for fold in info["folds"] if fold["test_end"] <= "20241231"]
    if [fold["fold_id"] for fold in folds] != ["fold2022", "fold2023", "fold2024"]:
        raise RuntimeError("wrong development folds")
    baseline = pd.read_parquet(BASE / "3d_oof.parquet", filters=[("trade_date", "<=", "20241231")])
    if baseline.trade_date.max() > "20241231":
        raise RuntimeError("confirmation read")
    OUT.mkdir(parents=True)
    base.dump_json(OUT / "preflight.json", {"contract_sha256": base.sha256_file(CONTRACT), "production_feature_columns": info["resolved"], "candidate_feature_columns": CANDIDATE_COLS, "folds": folds, "production_weight_config": info["metadata"].get("sample_weight_config"), "price_transform": "index OHLC / prior available index close - 1; stock qfq OHLC / same-row pre_close_qfq - 1", "confirmation_2025_read": False, "validation_2026_closed": True, "production_unchanged": True, "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"}})
    feature_db = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_db = duckdb.connect(str(base.LABEL_DB), read_only=True)
    rows: list[pd.DataFrame] = []
    try:
        for fold in folds:
            train = transformed_features(feature_db, fold["train_start"], fold["train_end"]).merge(base.query_labels(label_db, "executable_3d_open_return", fold["train_start"], fold["train_end"]), on=["trade_date", "stock_code"], how="inner", validate="one_to_one").dropna(subset=["target"]).reset_index(drop=True)
            test = transformed_features(feature_db, fold["test_start"], fold["test_end"]).merge(base.query_labels(label_db, "executable_3d_open_return", fold["test_start"], fold["mature_label_cutoff"]), on=["trade_date", "stock_code"], how="left", validate="one_to_one")
            model = xgb.XGBRegressor(**base.model_params(info["metadata"]))
            train_matrix, test_matrix = train[CANDIDATE_COLS].astype("float32"), test[CANDIDATE_COLS].astype("float32")
            model.fit(train_matrix, train.target.astype("float32"), sample_weight=base.daily_weights(train, info["metadata"].get("sample_weight_config")), verbose=False)
            first, second, third = (model.predict(test_matrix).astype("float64") for _ in range(3))
            if not (np.array_equal(first, second) and np.array_equal(first, third)):
                raise RuntimeError(f"{fold['fold_id']}: deterministic failure")
            scored = test[["trade_date", "stock_code", "target"]].copy()
            scored["pred_prob"], scored["fold_id"], scored["label_mature_within_dev"] = first, fold["fold_id"], scored.trade_date <= fold["mature_label_cutoff"]
            if scored.duplicated(["trade_date", "stock_code"]).any() or scored.stock_code.astype(str).str.endswith(".BJ").any() or scored.pred_prob.isna().any() or not np.isfinite(scored.pred_prob).all():
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
    gate = {"aggregate_rank_ic_not_weaker": delta["rank_ic_mean"] >= 0, "each_fold_rank_ic_not_weaker": all(item["delta"]["rank_ic_mean"] >= 0 for item in comparisons), "aggregate_top1_top3_top5_top10_not_weaker": all(delta[f"top{count}_excess_mean"] >= 0 for count in (1, 3, 5, 10)), "each_fold_top10_not_weaker": all(item["delta"]["top10_excess_mean"] >= 0 for item in comparisons)}
    summary = {"candidate_id": contract["candidate_id"], "stage": "development_2022_2024_only", "same_key_rows": len(candidate_eval), "fold_comparisons": comparisons, "aggregate_baseline": before, "aggregate_candidate": after, "aggregate_delta": delta, "acceptance_gate": gate, "decision": "development_pass_waiting_for_2025_confirmation" if all(gate.values()) else "reject_no_further_search", "confirmation_2025_read": False, "validation_2026_closed": True, "production_unchanged": True, "allow_next_layer_continue": False}
    base.dump_json(OUT / "evaluation_summary.json", summary)
    base.dump_json(OUT / "audit_handoff.json", {"status": summary["decision"], "candidate_oof_sha256": base.sha256_file(OUT / "candidate_oof.parquet"), "summary_sha256": base.sha256_file(OUT / "evaluation_summary.json"), "contract_sha256": base.sha256_file(CONTRACT), "confirmation_2025_read": False, "allow_next_layer_continue": False, "production_unchanged": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
