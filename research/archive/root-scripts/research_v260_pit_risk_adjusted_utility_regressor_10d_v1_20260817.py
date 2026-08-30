"""Frozen single-candidate research build for risk-adjusted 10D utility.

This is deliberately research-only.  It reads official 2021-2024 sessions,
never reads 2025/2026 rows, and never modifies production assets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

from research_v260_pit_oof_excess_logit_10d_v1_20260816 import (
    DEVELOPMENT_END,
    FEATURE_DB,
    FEATURE_TABLE,
    METADATA_PATH,
    WARMUP_START,
    canonical_frame_hash,
    feature_aliases,
    fit_baseline,
    json_dump,
    label_end_map,
    load_calendar,
    production_params,
    quote,
    sha256_bytes,
    sha256_file,
)


CONTRACT = Path(
    "quant/data_file/runtime/agent_workspaces/research-agent/work/"
    "v260_risk_adjusted_utility_regression_contract_20260817_r1/"
    "executable_training_contract_v1.json"
)
L2_DB = Path("quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb")
L2_TABLE = "STOCK_DAILY_DATA"
OUT = "quant/data_file/reports/model_agent_v260_pit_risk_adjusted_utility_regressor_10d_v1_20260817_r1"
TRAIN_START = "20220104"


def robust_features(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the frozen same-date robust transform without cross-date state."""
    out = frame.copy()
    small_days: dict[str, int] = {}
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce").astype("float64")
        grouped = values.groupby(out["trade_date"], sort=False)
        finite_count = grouped.transform(lambda item: int(np.isfinite(item).sum()))
        median = grouped.transform("median")
        mad = (values - median).abs().groupby(out["trade_date"], sort=False).transform("median")
        scale = 1.4826 * mad
        transformed = (values - median) / scale
        transformed = transformed.clip(-5.0, 5.0)
        transformed = transformed.where(scale > 1e-12, 0.0)
        transformed = transformed.where(np.isfinite(values), np.nan)
        transformed = transformed.where(finite_count >= 30, np.nan)
        out[column] = transformed.astype("float32")
        for date in out.loc[finite_count < 30, "trade_date"].unique():
            small_days[str(date)] = small_days.get(str(date), 0) + 1
    return out, small_days


def load_l2_label_state(calendar: list[str]) -> pd.DataFrame:
    """Materialize only the contract's open-price label and prior-risk state."""
    connection = duckdb.connect(str(L2_DB), read_only=True)
    try:
        prices = connection.execute(
            f"SELECT trade_date, stock_code, open FROM {quote(L2_TABLE)} "
            "WHERE trade_date BETWEEN ? AND ? AND stock_code NOT LIKE '%.BJ' "
            "ORDER BY stock_code, trade_date",
            [WARMUP_START, DEVELOPMENT_END],
        ).fetchdf()
    finally:
        connection.close()
    prices["trade_date"] = prices["trade_date"].astype(str).str.replace("-", "", regex=False)
    prices["stock_code"] = prices["stock_code"].astype(str)
    prices["open"] = pd.to_numeric(prices["open"], errors="coerce")
    if prices.duplicated(["trade_date", "stock_code"]).any() or prices["stock_code"].str.endswith(".BJ").any():
        raise RuntimeError("blocked_fail_closed_l2_key_or_no_bj")

    previous = {calendar[index]: calendar[index - 1] for index in range(1, len(calendar))}
    next_day = {calendar[index]: calendar[index + 1] for index in range(len(calendar) - 1)}
    plus12 = label_end_map(calendar)
    open_index = prices.set_index(["trade_date", "stock_code"])["open"]
    keys_previous = pd.MultiIndex.from_arrays([prices["trade_date"].map(previous), prices["stock_code"]])
    prices["daily_open_to_open_return"] = prices["open"] / open_index.reindex(keys_previous).to_numpy() - 1.0
    prices["risk_sigma"] = (
        prices.groupby("stock_code", sort=False)["daily_open_to_open_return"]
        .transform(lambda item: item.pow(2).rolling(20, min_periods=15).mean().shift(1))
        .pow(0.5)
        * np.sqrt(12.0)
    )

    result = prices[["trade_date", "stock_code", "risk_sigma"]].copy()
    result["post_open"] = open_index.reindex(
        pd.MultiIndex.from_arrays([result["trade_date"].map(next_day), result["stock_code"]])
    ).to_numpy()
    result["post12_open"] = open_index.reindex(
        pd.MultiIndex.from_arrays([result["trade_date"].map(plus12), result["stock_code"]])
    ).to_numpy()
    result["label_end_date"] = result["trade_date"].map(plus12)
    return result


def make_utility_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = frame.copy()
    out["executable_10d_open_return"] = (
        out["post12_open"] * (1.0 - 0.0003 - 0.0005 - 0.001)
        / (out["post_open"] * (1.0 + 0.0003 + 0.001))
        - 1.0
    )
    valid_sigma = out["risk_sigma"].where(np.isfinite(out["risk_sigma"]))
    date_floor = valid_sigma.groupby(out["trade_date"], sort=False).transform("median") * 0.25
    out["risk_denom"] = np.maximum(valid_sigma, date_floor)
    valid = np.isfinite(out[["executable_10d_open_return", "risk_denom"]]).all(axis=1)
    counts = valid.groupby(out["trade_date"], sort=False).transform("sum")
    out = out.loc[valid & (counts >= 30)].copy()
    median_return = out.groupby("trade_date", sort=False)["executable_10d_open_return"].transform("median")
    out["pit_continuous_10d_net_risk_adjusted_utility"] = (
        (out["executable_10d_open_return"] - median_return) / out["risk_denom"]
    ).clip(-5.0, 5.0)
    if not np.isfinite(out["pit_continuous_10d_net_risk_adjusted_utility"]).all():
        raise RuntimeError("blocked_fail_closed_nonfinite_utility_label")
    exclusions = {
        "label_rows_excluded": int(len(frame) - len(out)),
        "remaining_label_rows": int(len(out)),
    }
    return out, exclusions


def load_data(features: list[str], calendar: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    connection = duckdb.connect(str(FEATURE_DB), read_only=True)
    try:
        selected = ", ".join(quote(item) for item in features)
        source = connection.execute(
            f"SELECT trade_date, stock_code, {selected} FROM {quote(FEATURE_TABLE)} "
            "WHERE trade_date BETWEEN ? AND ? AND stock_code NOT LIKE '%.BJ' "
            "ORDER BY trade_date, stock_code",
            [WARMUP_START, DEVELOPMENT_END],
        ).fetchdf()
    finally:
        connection.close()
    source["trade_date"] = source["trade_date"].astype(str).str.replace("-", "", regex=False)
    source["stock_code"] = source["stock_code"].astype(str)
    if source.empty or source["trade_date"].max() > DEVELOPMENT_END:
        raise RuntimeError("blocked_fail_closed_feature_date_boundary")
    if source.duplicated(["trade_date", "stock_code"]).any() or source["stock_code"].str.endswith(".BJ").any():
        raise RuntimeError("blocked_fail_closed_feature_key_or_no_bj")
    source, small_days = robust_features(source, features)
    state = load_l2_label_state(calendar)
    merged = source.merge(state, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    result, exclusion = make_utility_labels(merged)
    return result, {"feature_small_cross_section_entries": int(sum(small_days.values())), **exclusion}


def ordered_top10(group: pd.DataFrame, score: str) -> set[str]:
    return set(
        group.sort_values([score, "stock_code"], ascending=[False, True], kind="mergesort")
        .head(10)["stock_code"].astype(str)
    )


def daily_spearman(frame: pd.DataFrame, score: str, label: str) -> tuple[float, int]:
    values: list[float] = []
    for _, group in frame.groupby("trade_date", sort=True):
        if len(group) < 30 or group[score].nunique() < 2 or group[label].nunique() < 2:
            continue
        values.append(float(group[score].rank(method="average").corr(group[label].rank(method="average"))))
    if not values:
        raise RuntimeError("blocked_fail_closed_empty_metric_spearman_domain")
    return float(np.mean(values)), len(values)


def top10_mean(frame: pd.DataFrame, score: str, value: str) -> tuple[float, list[set[str]]]:
    daily: list[float] = []
    sets: list[set[str]] = []
    for _, group in frame.groupby("trade_date", sort=True):
        if len(group) < 30:
            continue
        top = group.sort_values([score, "stock_code"], ascending=[False, True], kind="mergesort").head(10)
        daily.append(float(top[value].mean()))
        sets.append(set(top["stock_code"].astype(str)))
    if len(daily) < 2:
        raise RuntimeError("blocked_fail_closed_empty_metric_top10_domain")
    return float(np.mean(daily)), sets


def turnover(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        raise RuntimeError("blocked_fail_closed_turnover_domain")
    return float(np.mean([1.0 - len(sets[index] & sets[index - 1]) / 10.0 for index in range(1, len(sets))]))


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    base_ret, days = daily_spearman(frame, "baseline_oof_score_current", "executable_10d_open_return")
    cand_ret, candidate_days = daily_spearman(frame, "candidate_raw_score", "executable_10d_open_return")
    base_utility, utility_days = daily_spearman(frame, "baseline_oof_score_current", "pit_continuous_10d_net_risk_adjusted_utility")
    cand_utility, candidate_utility_days = daily_spearman(frame, "candidate_raw_score", "pit_continuous_10d_net_risk_adjusted_utility")
    if len({days, candidate_days, utility_days, candidate_utility_days}) != 1:
        raise RuntimeError("blocked_fail_closed_metric_day_domain_mismatch")
    base_top_ret, base_sets = top10_mean(frame, "baseline_oof_score_current", "executable_10d_open_return")
    cand_top_ret, candidate_sets = top10_mean(frame, "candidate_raw_score", "executable_10d_open_return")
    base_top_utility, _ = top10_mean(frame, "baseline_oof_score_current", "pit_continuous_10d_net_risk_adjusted_utility")
    cand_top_utility, _ = top10_mean(frame, "candidate_raw_score", "pit_continuous_10d_net_risk_adjusted_utility")
    return {
        "metric_days": days,
        "baseline_return_spearman": base_ret,
        "candidate_return_spearman": cand_ret,
        "baseline_utility_spearman": base_utility,
        "candidate_utility_spearman": cand_utility,
        "baseline_top10_net_return": base_top_ret,
        "candidate_top10_net_return": cand_top_ret,
        "baseline_top10_utility": base_top_utility,
        "candidate_top10_utility": cand_top_utility,
        "baseline_turnover_proxy": turnover(base_sets),
        "candidate_turnover_proxy": turnover(candidate_sets),
    }


def run_replay(
    replay_id: int,
    data: pd.DataFrame,
    features: list[str],
    ends: dict[str, str],
    base_params: dict[str, Any],
    candidate_params: dict[str, Any],
    folds: list[dict[str, str]],
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay_dir = root / "replays" / f"replay{replay_id}"
    replay_dir.mkdir(parents=True, exist_ok=False)
    rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    model_hashes: list[str] = []
    label = "pit_continuous_10d_net_risk_adjusted_utility"
    for fold in folds:
        train = data.loc[
            (data["trade_date"] >= TRAIN_START)
            & (data["trade_date"] < fold["test_start"])
            & (data["label_end_date"] < fold["test_start"])
        ].copy()
        test = data.loc[
            (data["trade_date"] >= fold["test_start"])
            & (data["trade_date"] <= fold["test_end"])
            & data["label_end_date"].notna()
        ].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"blocked_fail_closed_empty_fold_{fold['fold_id']}")
        baseline, baseline_hash = fit_baseline(train.rename(columns={label: "target"}), test, features, base_params)
        model = xgb.XGBRegressor(**candidate_params)
        model.fit(train[features].astype("float32"), train[label].astype("float32"), verbose=False)
        candidate = model.predict(test[features].astype("float32")).astype("float64")
        emitted = test[["trade_date", "stock_code", "executable_10d_open_return", label]].copy()
        emitted["fold_id"] = fold["fold_id"]
        emitted["baseline_oof_score_current"] = baseline
        emitted["candidate_raw_score"] = candidate
        emitted = emitted.sort_values(["trade_date", "stock_code"], kind="mergesort")
        if emitted.duplicated(["trade_date", "stock_code"]).any() or not np.isfinite(emitted[["baseline_oof_score_current", "candidate_raw_score"]]).all().all():
            raise RuntimeError("blocked_fail_closed_oof_key_or_finite")
        current = metrics(emitted)
        candidate_hash = sha256_bytes(model.get_booster().save_raw(raw_format="json"))
        current.update({
            "fold_id": fold["fold_id"],
            "train_rows": int(len(train)),
            "test_rows": int(len(emitted)),
            "baseline_model_sha256": baseline_hash,
            "candidate_model_sha256": candidate_hash,
        })
        fold_metrics.append(current)
        model_hashes.extend([baseline_hash, candidate_hash])
        rows.append(emitted)
    output = pd.concat(rows, ignore_index=True).sort_values(["trade_date", "stock_code"], kind="mergesort")
    if output.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("blocked_fail_closed_aggregate_duplicate_key")
    aggregate = metrics(output)
    checks = [aggregate, *fold_metrics]
    gates = {
        "same_key_domain": True,
        "pit_oof_integrity": True,
        "finite_output": bool(np.isfinite(output[["baseline_oof_score_current", "candidate_raw_score"]]).all().all()),
        "core_prediction_date_spearman_not_worse": all(item["candidate_return_spearman"] >= item["baseline_return_spearman"] for item in checks),
        "core_prediction_utility_spearman_not_worse": all(item["candidate_utility_spearman"] >= item["baseline_utility_spearman"] for item in checks),
        "top10_front_quality_net_return_not_worse": all(item["candidate_top10_net_return"] >= item["baseline_top10_net_return"] for item in checks),
        "top10_front_quality_utility_not_worse": all(item["candidate_top10_utility"] >= item["baseline_top10_utility"] for item in checks),
        "turnover_proxy_not_worse": all(item["candidate_turnover_proxy"] <= item["baseline_turnover_proxy"] for item in checks),
        "nonzero_effective_score_change": all(
            not np.array_equal(
                item_frame["baseline_oof_score_current"].rank(method="first").to_numpy(),
                item_frame["candidate_raw_score"].rank(method="first").to_numpy(),
            )
            for item_frame in [output, *[output.loc[output["fold_id"] == fold["fold_id"]] for fold in folds]]
        ),
    }
    hashes = {
        "baseline_oof_score_sha256": canonical_frame_hash(output, ["trade_date", "stock_code", "baseline_oof_score_current"]),
        "candidate_raw_score_sha256": canonical_frame_hash(output, ["trade_date", "stock_code", "candidate_raw_score"]),
        "model_binary_sha256": sha256_bytes("".join(model_hashes).encode("ascii")),
    }
    summary = {"replay_id": replay_id, "aggregate": aggregate, "fold_metrics": fold_metrics, "gates": gates, "hashes": hashes}
    json_dump(replay_dir / "replay_summary.json", summary)
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUT)
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists():
        raise RuntimeError(f"refusing_existing_output_directory:{root}")
    root.mkdir(parents=True)
    try:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        if (
            contract["candidate_id"] != "v260_pit_risk_adjusted_utility_regressor_10d_v1"
            or contract["candidate_count"] != 1
            or contract["budget"] != 1
        ):
            raise RuntimeError("blocked_fail_closed_contract_identity_or_budget")
        if xgb.__version__ != contract["loss_and_model"]["xgboost_version"]["required_exact_version"]:
            raise RuntimeError("blocked_fail_closed_xgboost_version")
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        connection = duckdb.connect(str(FEATURE_DB), read_only=True)
        try:
            available = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({quote(FEATURE_TABLE)})").fetchall()}
        finally:
            connection.close()
        requested, features, aliases = feature_aliases(metadata, available)
        raw_features = contract["input_representation"]["raw_features_fixed_order"]
        if requested != raw_features or len(features) != 40:
            raise RuntimeError("blocked_fail_closed_frozen_production_feature_order")
        calendar = load_calendar()
        ends = label_end_map(calendar)
        data, data_quality = load_data(features, calendar)
        if data["trade_date"].max() > DEVELOPMENT_END or (data["trade_date"] >= "20250101").any():
            raise RuntimeError("blocked_fail_closed_2025_2026_read_boundary")
        json_dump(root / "preflight.json", {
            "candidate_id": contract["candidate_id"],
            "contract_sha256": sha256_file(CONTRACT),
            "research_only": True,
            "approval_status": "research_only_not_for_l5",
            "date_reads": {"warmup": [WARMUP_START, "20211231"], "development": [TRAIN_START, DEVELOPMENT_END], "sealed": ["20250101+", "not_read"]},
            "inputs": {"feature_db": str(FEATURE_DB), "l2_db": str(L2_DB), "official_calendar_only": True, "metadata_sha256": sha256_file(METADATA_PATH), "requested_features": requested, "resolved_features": features, "qfq_alias_pairs": aliases},
            "input_rows": int(len(data)), "no_bj": True, "duplicate_key_groups": 0, "quality": data_quality,
        })
        params = dict(contract["loss_and_model"]["fixed_params"])
        params["missing"] = np.nan
        replays = [run_replay(index, data, features, ends, production_params(metadata), params, contract["fixed_folds"]["folds"], root) for index in (1, 2, 3)]
        output, first = replays[0]
        deterministic = all((summary["hashes"], summary["aggregate"], summary["fold_metrics"], summary["gates"]) == (first["hashes"], first["aggregate"], first["fold_metrics"], first["gates"]) for _, summary in replays[1:])
        gates = dict(first["gates"])
        gates["deterministic_replay_3_of_3"] = deterministic
        passed = all(gates.values())
        decision = "completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search"
        output.to_parquet(root / "baseline_candidate_same_key_oof.parquet", index=False)
        json_dump(root / "research_candidate_manifest.json", {
            "candidate_id": contract["candidate_id"], "asset_role": "l4_research_prediction_asset", "approval_status": "research_only_not_for_l5", "decision": decision, "production_unchanged": True, "allow_next_layer_continue": False,
        })
        json_dump(root / "evaluation_summary.json", {
            "candidate_id": contract["candidate_id"], "contract_sha256": sha256_file(CONTRACT), "decision": decision,
            "aggregate": first["aggregate"], "fold_metrics": first["fold_metrics"], "gates": gates,
            "deterministic": {"passed": deterministic, "replay_hashes": [summary["hashes"] for _, summary in replays]},
            "same_key_rows": int(len(output)), "ready_for_audit_review": passed, "allow_next_layer_continue": False,
            "sealed_2025_2026_not_read": True, "production_unchanged": True,
        })
        json_dump(root / "hash_inventory.json", {
            "script_sha256": sha256_file(Path(__file__)), "contract_sha256": sha256_file(CONTRACT),
            "preflight_sha256": sha256_file(root / "preflight.json"), "oof_sha256": sha256_file(root / "baseline_candidate_same_key_oof.parquet"),
            "evaluation_summary_sha256": sha256_file(root / "evaluation_summary.json"),
        })
        return 0
    except Exception as error:
        json_dump(root / "blocked_or_rejected.json", {"candidate_id": "v260_pit_risk_adjusted_utility_regressor_10d_v1", "status": "blocked_fail_closed", "error": str(error), "production_unchanged": True, "allow_next_layer_continue": False})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
