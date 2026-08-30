"""One-time sealed 2025 confirmation for the frozen 10D daily-zscore candidate."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

import build_expanding_pit_oof_baselines_20260829 as base
from build_10d_daily_zscore_target_dev2022_2024_20260830 import daily_zscore, metrics


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
PARENT = DATA_DIR / "reports" / "model_agent_10d_daily_zscore_target_dev2022_2024_20260830"
DEVELOPMENT = PARENT / "build_r1"
OUT = PARENT / "confirmation_2025_r1"
CONTRACT = PARENT / "confirmation_contract_2025.json"
BASE = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing second confirmation: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    development = json.loads((DEVELOPMENT / "evaluation_summary.json").read_text(encoding="utf-8"))
    if (
        contract["candidate_id"] != development["candidate_id"]
        or development["decision"] != "development_pass_waiting_for_2025_confirmation"
        or not all(development["acceptance_gate"].values())
        or development["confirmation_2025_read"]
        or not contract["single_confirmation"]
        or not contract["validation_2026_closed"]
    ):
        raise RuntimeError("development freeze or confirmation contract invalid")
    dates = base.read_open_dates()
    _, prepared = base.preflight(dates)
    info = prepared["10d"]
    folds = [fold for fold in info["folds"] if fold["fold_id"] == "fold2025"]
    if len(folds) != 1 or folds[0]["test_start"] < "20250101" or folds[0]["test_end"] != "20251231":
        raise RuntimeError("wrong confirmation fold")
    fold = folds[0]
    baseline = pd.read_parquet(BASE / "10d_oof.parquet", filters=[("trade_date", ">=", "20250101"), ("trade_date", "<=", "20251231")])
    if baseline.trade_date.min() < "20250101" or baseline.trade_date.max() > "20251231":
        raise RuntimeError("confirmation boundary failure")

    OUT.mkdir(parents=True)
    base.dump_json(
        OUT / "preflight.json",
        {
            "confirmation_contract_sha256": base.sha256_file(CONTRACT),
            "development_contract_sha256": base.sha256_file(PARENT / "training_contract.json"),
            "development_oof_sha256": base.sha256_file(DEVELOPMENT / "candidate_oof.parquet"),
            "development_summary_sha256": base.sha256_file(DEVELOPMENT / "evaluation_summary.json"),
            "fold": fold,
            "confirmation_2025_read": True,
            "validation_2026_read": False,
            "production_unchanged": True,
            "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
        },
    )

    feature_db = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_db = duckdb.connect(str(base.LABEL_DB), read_only=True)
    try:
        train = base.query_features(feature_db, info["resolved"], fold["train_start"], fold["train_end"]).merge(
            base.query_labels(label_db, "executable_10d_open_return", fold["train_start"], fold["train_end"]),
            on=["trade_date", "stock_code"], how="inner", validate="one_to_one"
        ).dropna(subset=["target"]).reset_index(drop=True)
        test = base.query_features(feature_db, info["resolved"], fold["test_start"], fold["test_end"]).merge(
            base.query_labels(label_db, "executable_10d_open_return", fold["test_start"], fold["mature_label_cutoff"]),
            on=["trade_date", "stock_code"], how="left", validate="one_to_one"
        )
        weights = base.daily_weights(train, info["metadata"].get("sample_weight_config"))
        params = base.model_params(info["metadata"])
        matrix_train = train[info["resolved"]].astype("float32")
        matrix_test = test[info["resolved"]].astype("float32")
        model = xgb.XGBRegressor(**params)
        model.fit(matrix_train, daily_zscore(train).astype("float32"), sample_weight=weights, verbose=False)
        first = model.predict(matrix_test).astype("float64")
        second = model.predict(matrix_test).astype("float64")
        third = model.predict(matrix_test).astype("float64")
    finally:
        feature_db.close()
        label_db.close()
    if not (np.array_equal(first, second) and np.array_equal(first, third)):
        raise RuntimeError("deterministic replay failure")
    candidate = test[["trade_date", "stock_code", "target"]].copy()
    candidate["pred_prob"] = first
    candidate["fold_id"] = fold["fold_id"]
    candidate["label_mature_within_dev"] = candidate.trade_date <= fold["mature_label_cutoff"]
    if (
        candidate.duplicated(["trade_date", "stock_code"]).any()
        or candidate.stock_code.astype(str).str.endswith(".BJ").any()
        or candidate.pred_prob.isna().any()
        or not np.isfinite(candidate.pred_prob).all()
    ):
        raise RuntimeError("candidate quality failure")
    candidate.to_parquet(OUT / "candidate_oof.parquet", index=False)
    baseline_eval = baseline.loc[baseline.label_mature_within_dev, ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    candidate_eval = candidate.loc[candidate.label_mature_within_dev, ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    if not baseline_eval[["trade_date", "stock_code"]].equals(candidate_eval[["trade_date", "stock_code"]]):
        raise RuntimeError("same key failure")
    before = metrics(baseline_eval)
    after = metrics(candidate_eval)
    delta = {key: after[key] - before[key] for key in before}
    gate = {
        "rank_ic_not_weaker": delta["rank_ic_mean"] >= 0,
        "top1_top3_top5_top10_not_weaker": all(delta[f"top{count}_excess_mean"] >= 0 for count in (1, 3, 5, 10)),
        "same_key_finite_no_bj_no_duplicate": True,
        "deterministic_replay_3_of_3": True,
    }
    summary = {
        "candidate_id": contract["candidate_id"],
        "stage": "sealed_confirmation_2025_only",
        "same_key_rows": len(candidate_eval),
        "baseline": before,
        "candidate": after,
        "delta": delta,
        "acceptance_gate": gate,
        "decision": "confirmed_ready_for_audit" if all(gate.values()) else "reject_no_further_search",
        "development_evidence_frozen": True,
        "confirmation_2025_read": True,
        "validation_2026_read": False,
        "production_unchanged": True,
        "allow_next_layer_continue": False,
    }
    base.dump_json(OUT / "evaluation_summary.json", summary)
    base.dump_json(
        OUT / "audit_handoff.json",
        {
            "status": summary["decision"],
            "candidate_oof_sha256": base.sha256_file(OUT / "candidate_oof.parquet"),
            "summary_sha256": base.sha256_file(OUT / "evaluation_summary.json"),
            "confirmation_contract_sha256": base.sha256_file(CONTRACT),
            "development_summary_sha256": base.sha256_file(DEVELOPMENT / "evaluation_summary.json"),
            "validation_2026_read": False,
            "allow_next_layer_continue": False,
            "production_unchanged": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
