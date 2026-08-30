"""Single frozen, research-only 1D rank-target candidate build."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

import build_expanding_pit_oof_baselines_20260829 as base


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
OUTPUT_DIR = DATA_DIR / "reports" / "model_agent_1d_cross_sectional_rank_target_20260829" / "build_r1"
CONTRACT_PATH = OUTPUT_DIR.parent / "training_contract.json"
BASELINE_ROOT = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"


def sha256(path: Path) -> str:
    return base.sha256_file(path)


def write_json(path: Path, data: object) -> None:
    base.dump_json(path, data)


def rank_target(frame: pd.DataFrame) -> pd.Series:
    # Grouping is strictly within observed training dates; test labels never
    # participate in this transform.
    return frame.groupby("trade_date", sort=False)["target"].rank(method="average", pct=True)


def fold_metrics(frame: pd.DataFrame) -> dict[str, float]:
    metrics, _ = base.evaluate(frame)
    return {key: float(metrics[key]) for key in ("rank_ic_mean", "top1_excess_mean", "top3_excess_mean", "top5_excess_mean", "top10_excess_mean")}


def main() -> int:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"refusing to overwrite candidate evidence: {OUTPUT_DIR}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["candidate_id"] != "expanding_pit_1d_cross_sectional_rank_target_v1":
        raise RuntimeError("unexpected candidate contract")
    if contract["validation_2026_closed"] is not True:
        raise RuntimeError("validation boundary is open")
    baseline = pd.read_parquet(BASELINE_ROOT / "1d_oof.parquet")
    if baseline["trade_date"].max() > base.DEVELOPMENT_END:
        raise RuntimeError("baseline includes forbidden dates")
    metadata = json.loads(base.HORIZONS[0].metadata_path.read_text(encoding="utf-8"))
    open_dates = base.read_open_dates()
    preflight, prepared = base.preflight(open_dates)
    info = prepared["1d"]
    if metadata != info["metadata"]:
        raise RuntimeError("production metadata drifted during candidate setup")
    OUTPUT_DIR.mkdir(parents=True)
    write_json(OUTPUT_DIR / "preflight.json", {
        "contract_path": str(CONTRACT_PATH), "contract_sha256": sha256(CONTRACT_PATH),
        "baseline_1d_oof_sha256": sha256(BASELINE_ROOT / "1d_oof.parquet"),
        "baseline_preflight_sha256": sha256(BASELINE_ROOT / "preflight.json"),
        "production_metadata_sha256": sha256(base.HORIZONS[0].metadata_path),
        "resolved_features": info["resolved"], "folds": info["folds"],
        "development_closed_after": base.DEVELOPMENT_END, "validation_2026_closed": True,
        "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
        "production_unchanged": True,
    })
    feature_connection = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(base.LABEL_DB), read_only=True)
    candidate_rows = []
    comparisons = []
    try:
        for fold in info["folds"]:
            train_x = base.query_features(feature_connection, info["resolved"], fold["train_start"], fold["train_end"])
            train_y = base.query_labels(label_connection, "executable_1d_open_return", fold["train_start"], fold["train_end"])
            train = train_x.merge(train_y, on=["trade_date", "stock_code"], how="inner", validate="one_to_one").dropna(subset=["target"]).copy()
            test_x = base.query_features(feature_connection, info["resolved"], fold["test_start"], fold["test_end"])
            test_y = base.query_labels(label_connection, "executable_1d_open_return", fold["test_start"], fold["mature_label_cutoff"])
            test = test_x.merge(test_y, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
            model = xgb.XGBRegressor(**base.model_params(metadata))
            y_rank = rank_target(train)
            model.fit(train[info["resolved"]].astype("float32"), y_rank.astype("float32"), verbose=False)
            first = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
            second = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
            third = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
            if not (np.array_equal(first, second) and np.array_equal(first, third)):
                raise RuntimeError(f"{fold['fold_id']}: deterministic replay failed")
            scored = test[["trade_date", "stock_code", "target"]].copy()
            scored["pred_prob"] = first
            scored["fold_id"] = fold["fold_id"]
            scored["label_mature_within_dev"] = scored["trade_date"] <= fold["mature_label_cutoff"]
            if scored.duplicated(["trade_date", "stock_code"]).any() or scored["stock_code"].str.endswith(".BJ").any() or scored["pred_prob"].isna().any() or not np.isfinite(scored["pred_prob"]).all():
                raise RuntimeError(f"{fold['fold_id']}: candidate quality gate failed")
            candidate_rows.append(scored)
        candidate = pd.concat(candidate_rows, ignore_index=True)
    finally:
        feature_connection.close()
        label_connection.close()
    base_frame = baseline[["trade_date", "stock_code", "target", "pred_prob", "fold_id", "label_mature_within_dev"]].copy()
    base_frame = base_frame.loc[base_frame["label_mature_within_dev"]].reset_index(drop=True)
    candidate_eval = candidate.loc[candidate["label_mature_within_dev"]].reset_index(drop=True)
    if not base_frame[["trade_date", "stock_code"]].equals(candidate_eval[["trade_date", "stock_code"]]):
        raise RuntimeError("same-key OOF closure failed")
    for fold_id in sorted(candidate_eval["fold_id"].unique()):
        left = fold_metrics(base_frame.loc[base_frame["fold_id"] == fold_id])
        right = fold_metrics(candidate_eval.loc[candidate_eval["fold_id"] == fold_id])
        comparisons.append({"fold_id": fold_id, "baseline": left, "candidate": right, "delta": {key: right[key] - left[key] for key in left}})
    aggregate_baseline = fold_metrics(base_frame)
    aggregate_candidate = fold_metrics(candidate_eval)
    aggregate_delta = {key: aggregate_candidate[key] - aggregate_baseline[key] for key in aggregate_baseline}
    gate = {
        "aggregate_rank_ic_not_weaker": aggregate_delta["rank_ic_mean"] >= 0,
        "each_fold_rank_ic_not_weaker": all(item["delta"]["rank_ic_mean"] >= 0 for item in comparisons),
        "aggregate_top1_top3_top5_top10_not_weaker": all(aggregate_delta[f"top{x}_excess_mean"] >= 0 for x in (1, 3, 5, 10)),
        "each_fold_top10_not_weaker": all(item["delta"]["top10_excess_mean"] >= 0 for item in comparisons),
    }
    passed = all(gate.values())
    candidate.to_parquet(OUTPUT_DIR / "candidate_oof.parquet", index=False)
    summary = {
        "candidate_id": contract["candidate_id"], "approval_status": "research_only_not_for_l5",
        "same_key_rows": int(len(candidate_eval)), "fold_comparisons": comparisons,
        "aggregate_baseline": aggregate_baseline, "aggregate_candidate": aggregate_candidate,
        "aggregate_delta": aggregate_delta, "acceptance_gate": gate,
        "decision": "completed_ready_for_audit" if passed else "reject_no_further_search",
        "ready_for_audit_review": True, "allow_next_layer_continue": False,
        "production_unchanged": True, "validation_2026_closed": True,
    }
    write_json(OUTPUT_DIR / "evaluation_summary.json", summary)
    write_json(OUTPUT_DIR / "audit_handoff.json", {
        "status": summary["decision"], "candidate_oof_sha256": sha256(OUTPUT_DIR / "candidate_oof.parquet"),
        "summary_sha256": sha256(OUTPUT_DIR / "evaluation_summary.json"),
        "contract_sha256": sha256(CONTRACT_PATH), "ready_for_audit_review": True,
        "allow_next_layer_continue": False, "production_unchanged": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
