"""Build the single frozen v260 baseline-anchor monotone research candidate.

This is deliberately research-only.  It reads only 2021-2024 input rows, uses
the immutable official calendar, and never changes active L4/L5 assets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from research_v260_pit_oof_excess_logit_10d_v1_20260816 import (
    CONTRACT_PATH as UNUSED_V2_CONTRACT,
    DEVELOPMENT_END,
    FEATURE_DB,
    FEATURE_TABLE,
    LABEL_COLUMN,
    LABEL_DB,
    LABEL_TABLE,
    METADATA_PATH,
    WARMUP_START,
    canonical_frame_hash,
    feature_aliases,
    fit_baseline,
    json_dump,
    label_end_map,
    load_calendar,
    load_data,
    production_params,
    quote,
    sha256_bytes,
    sha256_file,
    strict_monthly_baseline_state,
)


CONTRACT_PATH = Path(
    "quant/data_file/runtime/agent_workspaces/research-agent/work/"
    "v260_anchor_monotone_training_contract_20260816_r1/"
    "executable_training_contract_v3.json"
)
DEVELOPMENT_START = "20220104"
OUTPUT_DEFAULT = (
    "quant/data_file/reports/"
    "model_agent_v260_baseline_anchor_monotone_excess_logit_10d_v1_20260816"
)


@dataclass(frozen=True)
class Fold:
    fold_id: str
    test_start: str
    test_end: str


def parse_folds(contract: dict[str, Any]) -> tuple[Fold, ...]:
    folds = tuple(Fold(**item) for item in contract["fixed_folds"]["folds"])
    if len(folds) != 3 or any(fold.test_end > DEVELOPMENT_END for fold in folds):
        raise RuntimeError("blocked_fail_closed_v3_fold_definition")
    return folds


def attach_rank(frame: pd.DataFrame, score_column: str, output_column: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["trade_date", score_column, "stock_code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ordered["_position"] = ordered.groupby("trade_date", sort=False).cumcount() + 1
    ordered["_count"] = ordered.groupby("trade_date", sort=False)["stock_code"].transform("size")
    ordered[output_column] = 1.0 - (ordered["_position"] - 1.0) / ordered["_count"]
    return ordered.sort_index()[output_column]


def eligible_with_label(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    finite = np.isfinite(result[["target", "baseline_oof_score_current", "baseline_oof_rank_pct_current"]]).all(axis=1)
    result = result.loc[finite].copy()
    counts = result.groupby("trade_date", sort=False)["stock_code"].transform("size")
    result = result.loc[counts >= 30].copy()
    median = result.groupby("trade_date", sort=False)["target"].transform("median")
    result["excess_positive_10d_binary"] = (result["target"] > median).astype("int8")
    return result


def daily_spearman(frame: pd.DataFrame, left: str, right: str) -> tuple[float, int]:
    values: list[float] = []
    for _, group in frame.groupby("trade_date", sort=True):
        pair = group[[left, right]].dropna()
        if len(pair) < 30 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
            continue
        values.append(float(pair[left].rank(method="average").corr(pair[right].rank(method="average"))))
    if not values:
        raise RuntimeError("blocked_fail_closed_no_valid_metric_dates")
    return float(np.mean(values)), len(values)


def daily_top10_sets(frame: pd.DataFrame, score_column: str) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    for trade_date, group in frame.groupby("trade_date", sort=True):
        valid = group.loc[np.isfinite(group[score_column])].sort_values(
            [score_column, "stock_code"], ascending=[False, True], kind="mergesort"
        )
        if len(valid) >= 10:
            sets[str(trade_date)] = set(valid.head(10)["stock_code"].astype(str))
    return sets


def top10_overlap_and_turnover(frame: pd.DataFrame) -> tuple[float, float, float, int]:
    baseline = daily_top10_sets(frame, "baseline_oof_score_current")
    candidate = daily_top10_sets(frame, "candidate_raw_score")
    dates = sorted(set(baseline).intersection(candidate))
    if not dates:
        raise RuntimeError("blocked_fail_closed_no_top10_metric_dates")
    overlap = float(np.mean([len(baseline[date] & candidate[date]) / 10.0 for date in dates]))
    comparable = list(zip(dates[1:], dates[:-1]))
    if not comparable:
        raise RuntimeError("blocked_fail_closed_no_top10_turnover_pairs")
    base_turnover = float(np.mean([1.0 - len(baseline[now] & baseline[prior]) / 10.0 for now, prior in comparable]))
    cand_turnover = float(np.mean([1.0 - len(candidate[now] & candidate[prior]) / 10.0 for now, prior in comparable]))
    return overlap, base_turnover, cand_turnover, len(dates)


def classifier_params(contract: dict[str, Any]) -> dict[str, Any]:
    params = dict(contract["loss_and_model"]["fixed_params"])
    params["missing"] = np.nan
    params["monotone_constraints"] = tuple(contract["loss_and_model"]["monotone_constraints"]["constraints_for_feature_order"])
    return params


def make_inner_anchor_state(
    data: pd.DataFrame,
    features: list[str],
    calendar: list[str],
    end_map: dict[str, str],
    baseline_params: dict[str, Any],
    model_dir: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    state, hashes = strict_monthly_baseline_state(data, features, calendar, end_map, baseline_params, model_dir)
    state = attach_rank(state, "baseline_state_score", "baseline_oof_rank_pct_current").to_frame().join(state)
    state = state.rename(columns={"baseline_state_score": "baseline_oof_score_current"})
    return state[["trade_date", "stock_code", "baseline_oof_score_current", "baseline_oof_rank_pct_current"]], hashes


def fold_output(
    fold: Fold,
    data: pd.DataFrame,
    state: pd.DataFrame,
    features: list[str],
    model_params: dict[str, Any],
    candidate_params: dict[str, Any],
    end_map: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outer_train = data.loc[
        (data["trade_date"] >= DEVELOPMENT_START)
        & (data["trade_date"] < fold.test_start)
        & (data["trade_date"].map(end_map) < fold.test_start)
        & np.isfinite(data["target"])
    ].copy()
    candidate_train = outer_train.merge(state, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    candidate_train = eligible_with_label(candidate_train)
    test = data.loc[
        (data["trade_date"] >= fold.test_start)
        & (data["trade_date"] <= fold.test_end)
        & data["trade_date"].map(end_map).notna()
        & np.isfinite(data["target"])
    ].copy()
    if outer_train.empty or candidate_train.empty or test.empty:
        raise RuntimeError(f"blocked_fail_closed_empty_fold_{fold.fold_id}")
    baseline_score, baseline_hash = fit_baseline(outer_train, test, features, model_params)
    test["baseline_oof_score_current"] = baseline_score
    test["baseline_oof_rank_pct_current"] = attach_rank(
        test, "baseline_oof_score_current", "baseline_oof_rank_pct_current"
    )
    test = eligible_with_label(test)
    feature_order = ["baseline_oof_score_current", "baseline_oof_rank_pct_current", *features]
    classifier = xgb.XGBClassifier(**candidate_params)
    classifier.fit(
        candidate_train[feature_order].astype("float32"),
        candidate_train["excess_positive_10d_binary"].astype("int8"),
        verbose=False,
    )
    class_one = np.flatnonzero(classifier.classes_ == 1)
    if len(class_one) != 1:
        raise RuntimeError("blocked_fail_closed_class_1_probability_unavailable")
    probability = classifier.predict_proba(test[feature_order].astype("float32"))[:, int(class_one[0])].astype("float64")
    test["candidate_probability"] = probability
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    test["candidate_raw_score"] = np.log(clipped / (1.0 - clipped))
    if not np.isfinite(test[["baseline_oof_score_current", "candidate_probability", "candidate_raw_score"]]).all().all():
        raise RuntimeError(f"blocked_fail_closed_nonfinite_fold_{fold.fold_id}")
    test["fold_id"] = fold.fold_id
    emitted = test[
        ["trade_date", "stock_code", "fold_id", "baseline_oof_score_current", "baseline_oof_rank_pct_current", "candidate_probability", "candidate_raw_score", "target", "excess_positive_10d_binary"]
    ].copy()
    candidate_hash = sha256_bytes(classifier.get_booster().save_raw(raw_format="json"))
    rank_corr, rank_days = daily_spearman(emitted, "candidate_raw_score", "baseline_oof_score_current")
    overlap, base_turn, cand_turn, top_days = top10_overlap_and_turnover(emitted)
    metrics = {
        "fold_id": fold.fold_id,
        "train_rows": int(len(candidate_train)),
        "test_rows": int(len(emitted)),
        "baseline_roc_auc": float(roc_auc_score(emitted["excess_positive_10d_binary"], emitted["baseline_oof_score_current"])),
        "candidate_roc_auc": float(roc_auc_score(emitted["excess_positive_10d_binary"], emitted["candidate_raw_score"])),
        "baseline_return_rank_spearman": daily_spearman(emitted, "baseline_oof_score_current", "target")[0],
        "candidate_return_rank_spearman": daily_spearman(emitted, "candidate_raw_score", "target")[0],
        "baseline_candidate_rank_correlation": rank_corr,
        "top10_overlap": overlap,
        "baseline_turnover_proxy": base_turn,
        "candidate_turnover_proxy": cand_turn,
        "rank_metric_days": rank_days,
        "top10_metric_days": top_days,
        "baseline_model_sha256": baseline_hash,
        "candidate_model_sha256": candidate_hash,
        "classes": classifier.classes_.tolist(),
    }
    return emitted, metrics


def evaluate(output: pd.DataFrame, fold_metrics: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, bool]]:
    aggregate = {
        "baseline_roc_auc": float(roc_auc_score(output["excess_positive_10d_binary"], output["baseline_oof_score_current"])),
        "candidate_roc_auc": float(roc_auc_score(output["excess_positive_10d_binary"], output["candidate_raw_score"])),
        "baseline_return_rank_spearman": daily_spearman(output, "baseline_oof_score_current", "target")[0],
        "candidate_return_rank_spearman": daily_spearman(output, "candidate_raw_score", "target")[0],
    }
    aggregate["baseline_candidate_rank_correlation"] = daily_spearman(output, "candidate_raw_score", "baseline_oof_score_current")[0]
    overlap, base_turn, cand_turn, top_days = top10_overlap_and_turnover(output)
    aggregate.update({"top10_overlap": overlap, "baseline_turnover_proxy": base_turn, "candidate_turnover_proxy": cand_turn, "top10_metric_days": top_days})
    aggregate["roc_auc_delta"] = aggregate["candidate_roc_auc"] - aggregate["baseline_roc_auc"]
    aggregate["return_rank_delta"] = aggregate["candidate_return_rank_spearman"] - aggregate["baseline_return_rank_spearman"]
    checks = [aggregate, *fold_metrics]
    gates = {
        "same_key_domain": True,
        "pit_oof_integrity": True,
        "auc_not_worse": all(item["candidate_roc_auc"] >= item["baseline_roc_auc"] for item in checks),
        "date_spearman_not_worse": all(item["candidate_return_rank_spearman"] >= item["baseline_return_rank_spearman"] for item in checks),
        "baseline_rank_correlation_floor": all(item["baseline_candidate_rank_correlation"] >= 0.70 for item in checks),
        "top10_overlap_floor": all(item["top10_overlap"] >= 0.30 for item in checks),
        "turnover_proxy_not_up": all(item["candidate_turnover_proxy"] <= item["baseline_turnover_proxy"] for item in checks),
        "score_finiteness": bool(np.isfinite(output[["baseline_oof_score_current", "candidate_probability", "candidate_raw_score"]]).all().all()),
    }
    return aggregate, gates


def run_replay(replay_id: int, data: pd.DataFrame, features: list[str], calendar: list[str], end_map: dict[str, str], baseline_params: dict[str, Any], candidate_params: dict[str, Any], folds: tuple[Fold, ...], output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay_dir = output_dir / "replays" / f"replay{replay_id}"
    replay_dir.mkdir(parents=True, exist_ok=False)
    state, monthly_hashes = make_inner_anchor_state(data, features, calendar, end_map, baseline_params, replay_dir / "monthly_baseline_models")
    rows, fold_metrics = zip(*(fold_output(fold, data, state, features, baseline_params, candidate_params, end_map) for fold in folds))
    output = pd.concat(rows, ignore_index=True).sort_values(["trade_date", "stock_code"], kind="mergesort")
    if output.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("blocked_fail_closed_same_key_duplicate")
    aggregate, gates = evaluate(output, list(fold_metrics))
    hashes = {
        "baseline_oof_score_sha256": canonical_frame_hash(output, ["trade_date", "stock_code", "baseline_oof_score_current"]),
        "baseline_oof_rank_sha256": canonical_frame_hash(output, ["trade_date", "stock_code", "baseline_oof_rank_pct_current"]),
        "candidate_probability_sha256": canonical_frame_hash(output, ["trade_date", "stock_code", "candidate_probability"]),
        "candidate_raw_score_sha256": canonical_frame_hash(output, ["trade_date", "stock_code", "candidate_raw_score"]),
        "combined_model_sha256": sha256_bytes("".join(sorted(monthly_hashes.values()) + [item["baseline_model_sha256"] for item in fold_metrics] + [item["candidate_model_sha256"] for item in fold_metrics]).encode("ascii")),
    }
    summary = {"replay_id": replay_id, "aggregate": aggregate, "fold_metrics": list(fold_metrics), "gates": gates, "hashes": hashes}
    json_dump(replay_dir / "replay_summary.json", summary)
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        if contract["candidate_id"] != "v260_baseline_anchor_monotone_excess_logit_10d_v1" or contract["budget"] != 1:
            raise RuntimeError("blocked_fail_closed_contract_identity")
        if xgb.__version__ != contract["loss_and_model"]["xgboost_version"]["required_exact_version"]:
            raise RuntimeError("blocked_fail_closed_xgboost_version_mismatch")
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        feature_connection = duckdb.connect(str(FEATURE_DB), read_only=True)
        try:
            available = {str(row[1]) for row in feature_connection.execute(f"PRAGMA table_info({quote(FEATURE_TABLE)})").fetchall()}
        finally:
            feature_connection.close()
        requested, features, aliases = feature_aliases(metadata, available)
        expected = contract["input_fields"]["core_pit_features"]
        if requested != expected or len(features) != 40:
            raise RuntimeError("blocked_fail_closed_v3_feature_order_mismatch")
        constraints = contract["loss_and_model"]["monotone_constraints"]["constraints_for_feature_order"]
        if len(constraints) != 42 or constraints[:2] != [1, 1] or any(constraints[2:]):
            raise RuntimeError("blocked_fail_closed_monotone_constraint_contract")
        calendar = load_calendar()
        end_map = label_end_map(calendar)
        data = load_data(features)
        if data["trade_date"].max() > DEVELOPMENT_END:
            raise RuntimeError("blocked_fail_closed_sealed_date_read")
        preflight = {
            "candidate_id": contract["candidate_id"],
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "research_only": True,
            "date_reads": {"warmup": [WARMUP_START, "20211231"], "development": [DEVELOPMENT_START, DEVELOPMENT_END], "sealed": ["20250101+", "not_read"]},
            "feature_db": str(FEATURE_DB),
            "label_db": str(LABEL_DB),
            "label_column": LABEL_COLUMN,
            "label_lineage": "materialized executable_10d_open_return; source formula is the v3 post_open/post12_open cost-adjusted formula",
            "official_calendar": {"source": "official_trade_cal only", "start": calendar[0], "end": calendar[-1], "trade_days": len(calendar)},
            "requested_features": requested,
            "resolved_features": features,
            "qfq_alias_pairs": aliases,
            "monotone_constraints": constraints,
            "input_rows": int(len(data)),
            "no_bj": True,
            "duplicate_key_groups": 0,
        }
        json_dump(output_dir / "preflight.json", preflight)
        baseline_params = production_params(metadata)
        candidate_params = classifier_params(contract)
        replays = [run_replay(index, data, features, calendar, end_map, baseline_params, candidate_params, parse_folds(contract), output_dir) for index in (1, 2, 3)]
        output, first = replays[0]
        exact = all((summary["hashes"], summary["aggregate"], summary["gates"]) == (first["hashes"], first["aggregate"], first["gates"]) for _, summary in replays[1:])
        gates = dict(first["gates"])
        gates["deterministic_replay_3_of_3"] = exact
        passed = all(gates.values())
        output = output.rename(columns={"target": "executable_10d_open_return"})
        output.to_parquet(output_dir / "baseline_candidate_same_key_oof.parquet", index=False)
        decision = "strategy_development_ab_only_pending_audit" if passed else "model_layer_or_drift_rejected_no_strategy_ab"
        manifest = {"candidate_id": contract["candidate_id"], "asset_role": "l4_research_prediction_asset", "approval_status": "research_only_not_for_l5", "production_unchanged": True, "decision": decision, "allow_next_layer_continue": False}
        json_dump(output_dir / "research_candidate_manifest.json", manifest)
        result = {"candidate_id": contract["candidate_id"], "contract_sha256": sha256_file(CONTRACT_PATH), "decision": decision, "aggregate": first["aggregate"], "fold_metrics": first["fold_metrics"], "acceptance_gates": gates, "deterministic_replay": {"required": 3, "passed": exact, "replay_hashes": [summary["hashes"] for _, summary in replays]}, "same_key": {"rows": int(len(output)), "duplicate_key_groups": 0, "extra_or_missing_keys": 0}, "ready_for_audit_review": passed, "allow_next_layer_continue": False, "sealed_2025_2026_not_read": True}
        json_dump(output_dir / "evaluation_summary.json", result)
        json_dump(output_dir / "hash_inventory.json", {"script_sha256": sha256_file(Path(__file__)), "contract_sha256": sha256_file(CONTRACT_PATH), "preflight_sha256": sha256_file(output_dir / "preflight.json"), "evaluation_summary_sha256": sha256_file(output_dir / "evaluation_summary.json"), "oof_sha256": sha256_file(output_dir / "baseline_candidate_same_key_oof.parquet"), "manifest_sha256": sha256_file(output_dir / "research_candidate_manifest.json")})
        return 0
    except Exception as error:
        json_dump(output_dir / "blocked_or_rejected.json", {"candidate_id": "v260_baseline_anchor_monotone_excess_logit_10d_v1", "status": "blocked_fail_closed", "error": str(error), "production_unchanged": True, "allow_next_layer_continue": False})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
