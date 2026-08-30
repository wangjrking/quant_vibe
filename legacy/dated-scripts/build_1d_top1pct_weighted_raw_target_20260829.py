"""Build one strict pre-2026 PIT/OOF 1D front-weighted research candidate."""

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
OUTPUT_DIR = DATA_DIR / "reports" / "model_agent_1d_top1pct_weighted_raw_target_20260829" / "build_r2"
CONTRACT_PATH = OUTPUT_DIR.parent / "training_contract.json"
BASELINE_ROOT = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    result, _ = base.evaluate(frame)
    return {
        key: float(result[key])
        for key in ("rank_ic_mean", "top1_excess_mean", "top3_excess_mean", "top5_excess_mean", "top10_excess_mean")
    }


def daily_top1pct_weights(frame: pd.DataFrame, top_pct: float, multiplier: float) -> np.ndarray:
    frame = frame.reset_index(drop=True)
    weights = np.ones(len(frame), dtype=np.float32)
    for _, group in frame.groupby("trade_date", sort=False):
        top_count = max(1, int(np.ceil(len(group) * top_pct)))
        winners = group.nlargest(top_count, "target", keep="first").index
        weights[winners] = np.float32(multiplier)
    return weights


def main() -> int:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"refusing to overwrite candidate evidence: {OUTPUT_DIR}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    change = contract.get("single_change", {})
    if contract.get("candidate_id") != "expanding_pit_1d_daily_top1pct_weighted_raw_return_v1":
        raise RuntimeError("unexpected candidate contract")
    if contract.get("training_window_policy") != "expanding_available_history" or not contract["development_window"].get("validation_2026_closed"):
        raise RuntimeError("training-window or validation boundary violation")
    if change != {
        "kind": "daily_top1pct_raw_return_sample_weight",
        "target": "unchanged executable_1d_open_return raw return",
        "top_percentile": 0.01,
        "top_multiplier": 1.5,
        "other_multiplier": 1.0,
        "feature_columns": "unchanged production 1D resolved feature list",
        "xgboost_parameters": "unchanged production 1D specification",
    }:
        raise RuntimeError("contract change is not the frozen single change")
    baseline = pd.read_parquet(BASELINE_ROOT / "1d_oof.parquet")
    if baseline["trade_date"].max() > base.DEVELOPMENT_END:
        raise RuntimeError("baseline reads forbidden validation dates")
    open_dates = base.read_open_dates()
    _, prepared = base.preflight(open_dates)
    info = prepared["1d"]
    metadata = info["metadata"]
    OUTPUT_DIR.mkdir(parents=True)
    base.dump_json(OUTPUT_DIR / "preflight.json", {
        "candidate_id": contract["candidate_id"],
        "contract_sha256": base.sha256_file(CONTRACT_PATH),
        "baseline_oof_sha256": base.sha256_file(BASELINE_ROOT / "1d_oof.parquet"),
        "production_metadata_sha256": base.sha256_file(base.HORIZONS[0].metadata_path),
        "resolved_features": info["resolved"],
        "folds": info["folds"],
        "top_pct": change["top_percentile"],
        "top_multiplier": change["top_multiplier"],
        "development_closed_after": base.DEVELOPMENT_END,
        "validation_2026_closed": True,
        "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
        "production_unchanged": True,
    })
    feature_db = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_db = duckdb.connect(str(base.LABEL_DB), read_only=True)
    frames: list[pd.DataFrame] = []
    try:
        for fold in info["folds"]:
            train_x = base.query_features(feature_db, info["resolved"], fold["train_start"], fold["train_end"])
            train_y = base.query_labels(label_db, "executable_1d_open_return", fold["train_start"], fold["train_end"])
            train = train_x.merge(train_y, on=["trade_date", "stock_code"], how="inner", validate="one_to_one").dropna(subset=["target"]).reset_index(drop=True)
            test_x = base.query_features(feature_db, info["resolved"], fold["test_start"], fold["test_end"])
            test_y = base.query_labels(label_db, "executable_1d_open_return", fold["test_start"], fold["mature_label_cutoff"])
            test = test_x.merge(test_y, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
            model = xgb.XGBRegressor(**base.model_params(metadata))
            weights = daily_top1pct_weights(train, change["top_percentile"], change["top_multiplier"])
            model.fit(train[info["resolved"]].astype("float32"), train["target"].astype("float32"), sample_weight=weights, verbose=False)
            first = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
            second = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
            third = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
            if not (np.array_equal(first, second) and np.array_equal(first, third)):
                raise RuntimeError(f"{fold['fold_id']}: deterministic replay failed")
            scored = test[["trade_date", "stock_code", "target"]].copy()
            scored["pred_prob"] = first
            scored["fold_id"] = fold["fold_id"]
            scored["label_mature_within_dev"] = scored["trade_date"] <= fold["mature_label_cutoff"]
            if (scored.duplicated(["trade_date", "stock_code"]).any() or scored["stock_code"].astype(str).str.endswith(".BJ").any()
                    or scored["pred_prob"].isna().any() or not np.isfinite(scored["pred_prob"]).all()):
                raise RuntimeError(f"{fold['fold_id']}: candidate quality gate failed")
            frames.append(scored)
    finally:
        feature_db.close()
        label_db.close()
    candidate = pd.concat(frames, ignore_index=True)
    base_eval = baseline.loc[baseline["label_mature_within_dev"], ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    candidate_eval = candidate.loc[candidate["label_mature_within_dev"], ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    if not base_eval[["trade_date", "stock_code"]].equals(candidate_eval[["trade_date", "stock_code"]]):
        raise RuntimeError("same-key OOF closure failed")
    comparisons = []
    for fold_id in sorted(candidate_eval["fold_id"].unique()):
        left, right = metrics(base_eval.query("fold_id == @fold_id")), metrics(candidate_eval.query("fold_id == @fold_id"))
        comparisons.append({"fold_id": fold_id, "baseline": left, "candidate": right, "delta": {key: right[key] - left[key] for key in left}})
    left, right = metrics(base_eval), metrics(candidate_eval)
    delta = {key: right[key] - left[key] for key in left}
    gate = {
        "aggregate_rank_ic_not_weaker": delta["rank_ic_mean"] >= 0,
        "each_fold_rank_ic_not_weaker": all(item["delta"]["rank_ic_mean"] >= 0 for item in comparisons),
        "aggregate_top1_top3_top5_top10_not_weaker": all(delta[f"top{x}_excess_mean"] >= 0 for x in (1, 3, 5, 10)),
        "each_fold_top10_not_weaker": all(item["delta"]["top10_excess_mean"] >= 0 for item in comparisons),
    }
    candidate.to_parquet(OUTPUT_DIR / "candidate_oof.parquet", index=False)
    summary = {
        "candidate_id": contract["candidate_id"], "approval_status": "research_only_not_for_l5",
        "same_key_rows": int(len(candidate_eval)), "fold_comparisons": comparisons,
        "aggregate_baseline": left, "aggregate_candidate": right, "aggregate_delta": delta,
        "acceptance_gate": gate,
        "decision": "completed_ready_for_audit" if all(gate.values()) else "reject_no_further_search",
        "ready_for_audit_review": True, "allow_next_layer_continue": False,
        "production_unchanged": True, "validation_2026_closed": True,
    }
    base.dump_json(OUTPUT_DIR / "evaluation_summary.json", summary)
    base.dump_json(OUTPUT_DIR / "audit_handoff.json", {
        "status": summary["decision"],
        "candidate_oof_sha256": base.sha256_file(OUTPUT_DIR / "candidate_oof.parquet"),
        "summary_sha256": base.sha256_file(OUTPUT_DIR / "evaluation_summary.json"),
        "contract_sha256": base.sha256_file(CONTRACT_PATH),
        "ready_for_audit_review": True, "allow_next_layer_continue": False,
        "production_unchanged": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
