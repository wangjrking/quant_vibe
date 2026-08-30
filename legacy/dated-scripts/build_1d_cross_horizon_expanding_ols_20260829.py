"""Build one strict pre-2026 expanding OOF cross-horizon 1D meta candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_expanding_pit_oof_baselines_20260829 as base


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
OUTPUT_DIR = DATA_DIR / "reports" / "model_agent_1d_cross_horizon_expanding_ols_20260829" / "build_r1"
CONTRACT_PATH = OUTPUT_DIR.parent / "training_contract.json"
BASELINE_ROOT = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"
HORIZONS = ("1d", "3d", "5d", "10d")
FEATURES = tuple(f"rank_{h}" for h in HORIZONS)


def metric_subset(frame: pd.DataFrame) -> dict[str, float]:
    result, _ = base.evaluate(frame)
    return {key: float(result[key]) for key in ("rank_ic_mean", "top1_excess_mean", "top3_excess_mean", "top5_excess_mean", "top10_excess_mean")}


def read_scores() -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for horizon in HORIZONS:
        frame = pd.read_parquet(BASELINE_ROOT / f"{horizon}_oof.parquet", columns=["trade_date", "stock_code", "pred_prob"])
        if frame["trade_date"].max() > base.DEVELOPMENT_END:
            raise RuntimeError(f"{horizon}: forbidden validation-date read")
        frame = frame.rename(columns={"pred_prob": horizon})
        merged = frame if merged is None else merged.merge(frame, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    assert merged is not None
    for horizon in HORIZONS:
        merged[f"rank_{horizon}"] = merged.groupby("trade_date", sort=False)[horizon].rank(method="average", pct=True).astype("float64")
    return merged[["trade_date", "stock_code", *FEATURES]]


def fit_ols(history: pd.DataFrame) -> np.ndarray:
    design = np.column_stack([np.ones(len(history), dtype=np.float64), history.loc[:, FEATURES].to_numpy(dtype=np.float64)])
    target = history["target"].to_numpy(dtype=np.float64)
    return np.linalg.lstsq(design, target, rcond=None)[0]


def predict(features: pd.DataFrame, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features), dtype=np.float64), features.loc[:, FEATURES].to_numpy(dtype=np.float64)])
    return design @ coefficients


def main() -> int:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"refusing to overwrite candidate evidence: {OUTPUT_DIR}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    change = contract["single_change"]
    if contract["candidate_id"] != "expanding_pit_oof_cross_horizon_ols_1d_v1" or change["input_scores"] != list(HORIZONS):
        raise RuntimeError("frozen contract drift")
    if not contract["development_window"]["validation_2026_closed"] or change["no_hyperparameter_search"] is not True:
        raise RuntimeError("validation boundary or selection rule violation")
    one = pd.read_parquet(BASELINE_ROOT / "1d_oof.parquet")
    if one["trade_date"].max() > base.DEVELOPMENT_END:
        raise RuntimeError("baseline includes forbidden dates")
    scores = read_scores()
    if len(scores) != len(one):
        raise RuntimeError("base-score same-key closure failed")
    source = one[["trade_date", "stock_code", "target", "fold_id", "label_mature_within_dev"]].merge(scores, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    if len(source) != len(one):
        raise RuntimeError("source same-key closure failed")
    frames: list[pd.DataFrame] = []
    completed_history: list[pd.DataFrame] = []
    fold_models: list[dict[str, object]] = []
    for fold_id in sorted(source["fold_id"].unique()):
        current = source.loc[source["fold_id"] == fold_id].copy()
        # Score before adding current mature labels to later-fold history.
        if not completed_history:
            current["pred_prob"] = current["rank_1d"].to_numpy(dtype=np.float64)
            fold_models.append({"fold_id": fold_id, "mode": "cold_start_1d_rank", "training_source_folds": [], "coefficients": None})
        else:
            history = pd.concat(completed_history, ignore_index=True)
            coefficients = fit_ols(history)
            first = predict(current, coefficients)
            second = predict(current, coefficients)
            third = predict(current, coefficients)
            if not (np.array_equal(first, second) and np.array_equal(first, third)):
                raise RuntimeError(f"{fold_id}: deterministic replay 3/3 failed")
            current["pred_prob"] = first
            fold_models.append({
                "fold_id": fold_id,
                "mode": "expanding_ols",
                "training_source_folds": sorted(history["fold_id"].unique().tolist()),
                "training_rows": int(len(history)),
                "coefficients": {
                    "intercept": float(coefficients[0]),
                    **{feature: float(value) for feature, value in zip(FEATURES, coefficients[1:])},
                },
            })
        if current.duplicated(["trade_date", "stock_code"]).any() or current["stock_code"].astype(str).str.endswith(".BJ").any() or current["pred_prob"].isna().any() or not np.isfinite(current["pred_prob"]).all():
            raise RuntimeError(f"{fold_id}: candidate quality gate failed")
        frames.append(current[["trade_date", "stock_code", "target", "pred_prob", "fold_id", "label_mature_within_dev"]])
        completed_history.append(current.loc[current["label_mature_within_dev"], ["trade_date", "stock_code", "target", "fold_id", *FEATURES]].dropna(subset=["target"]).copy())
    candidate = pd.concat(frames, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True)
    candidate.to_parquet(OUTPUT_DIR / "candidate_oof.parquet", index=False)
    base_eval = one.loc[one["label_mature_within_dev"], ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    candidate_eval = candidate.loc[candidate["label_mature_within_dev"], ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]].reset_index(drop=True)
    if not base_eval[["trade_date", "stock_code"]].equals(candidate_eval[["trade_date", "stock_code"]]):
        raise RuntimeError("evaluated same-key closure failed")
    folds = []
    for fold_id in sorted(candidate_eval["fold_id"].unique()):
        left, right = metric_subset(base_eval.query("fold_id == @fold_id")), metric_subset(candidate_eval.query("fold_id == @fold_id"))
        folds.append({"fold_id": fold_id, "baseline": left, "candidate": right, "delta": {key: right[key] - left[key] for key in left}})
    left, right = metric_subset(base_eval), metric_subset(candidate_eval)
    delta = {key: right[key] - left[key] for key in left}
    gate = {"aggregate_rank_ic_not_weaker": delta["rank_ic_mean"] >= 0, "each_fold_rank_ic_not_weaker": all(item["delta"]["rank_ic_mean"] >= 0 for item in folds), "aggregate_top1_top3_top5_top10_not_weaker": all(delta[f"top{x}_excess_mean"] >= 0 for x in (1, 3, 5, 10)), "each_fold_top10_not_weaker": all(item["delta"]["top10_excess_mean"] >= 0 for item in folds)}
    summary = {"candidate_id": contract["candidate_id"], "approval_status": "research_only_not_for_l5", "same_key_rows": int(len(candidate_eval)), "fold_meta_models": fold_models, "fold_comparisons": folds, "aggregate_baseline": left, "aggregate_candidate": right, "aggregate_delta": delta, "acceptance_gate": gate, "decision": "completed_ready_for_audit" if all(gate.values()) else "reject_no_further_search", "ready_for_audit_review": True, "allow_next_layer_continue": False, "production_unchanged": True, "validation_2026_closed": True}
    base.dump_json(OUTPUT_DIR / "preflight.json", {"contract_sha256": base.sha256_file(CONTRACT_PATH), "baseline_1d_sha256": base.sha256_file(BASELINE_ROOT / "1d_oof.parquet"), "baseline_3d_sha256": base.sha256_file(BASELINE_ROOT / "3d_oof.parquet"), "baseline_5d_sha256": base.sha256_file(BASELINE_ROOT / "5d_oof.parquet"), "baseline_10d_sha256": base.sha256_file(BASELINE_ROOT / "10d_oof.parquet"), "development_closed_after": base.DEVELOPMENT_END, "validation_2026_closed": True, "production_unchanged": True, "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"}})
    base.dump_json(OUTPUT_DIR / "evaluation_summary.json", summary)
    base.dump_json(OUTPUT_DIR / "audit_handoff.json", {"status": summary["decision"], "candidate_oof_sha256": base.sha256_file(OUTPUT_DIR / "candidate_oof.parquet"), "summary_sha256": base.sha256_file(OUTPUT_DIR / "evaluation_summary.json"), "contract_sha256": base.sha256_file(CONTRACT_PATH), "ready_for_audit_review": True, "allow_next_layer_continue": False, "production_unchanged": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
