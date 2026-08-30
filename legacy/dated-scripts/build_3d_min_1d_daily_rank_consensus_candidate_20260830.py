"""Build one frozen 3D-and-1D consensus-label candidate under strict PIT.

The candidate keeps the approved 3D feature set, model parameters, expanding
history and evaluation target.  Its only changed semantic is a deterministic
training label: the smaller of same-day 3D and 1D cross-sectional percentile
ranks.  No 2025 or 2026 data, calendar, label, or metric is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

from build_expanding_pit_oof_baselines_20260829 import (
    CALENDAR_2021_2024,
    FEATURE_DB,
    FEATURE_TABLE,
    HORIZONS,
    LABEL_DB,
    LABEL_TABLE,
    POLICY_PATH,
    canonical_frame_hash,
    daily_weights,
    evaluate,
    model_params,
    query_features,
    query_labels,
    quote,
    sha256_file,
)
from incremental_formal_l4_duckdb_mainline import resolve_savedmodel_feature_aliases


CANDIDATE_ID = "v261_3d_min_1d_daily_rank_consensus_label_v1"
DEVELOPMENT_START = "20220101"
DEVELOPMENT_END = "20241231"
MATURITY_SESSIONS = 4
BASELINE_ROOT = Path("quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3")
DEFAULT_OUTPUT = Path("quant/data_file/reports/model_agent_3d_min_1d_daily_rank_consensus_20260830")


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def strict_open_dates() -> list[str]:
    connection = duckdb.connect(str(CALENDAR_2021_2024), read_only=True)
    try:
        dates = [str(row[0]) for row in connection.execute(
            "SELECT cal_date FROM official_trade_cal WHERE is_open = 1 AND cal_date <= ? ORDER BY cal_date",
            [DEVELOPMENT_END],
        ).fetchall()]
    finally:
        connection.close()
    if not dates or dates[0] > "20210104" or dates[-1] != DEVELOPMENT_END or dates != sorted(set(dates)):
        raise RuntimeError("blocked_fail_closed_official_pre2025_calendar")
    return dates


def session_before(dates: list[str], date: str, sessions: int) -> str:
    index = dates.index(date)
    if index < sessions:
        raise RuntimeError("blocked_fail_closed_calendar_history")
    return dates[index - sessions]


def folds(dates: list[str], train_start: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for year in ("2022", "2023", "2024"):
        all_dates = [date for date in dates if date.startswith(year)]
        if not all_dates:
            raise RuntimeError(f"blocked_fail_closed_calendar_no_{year}")
        test_start = all_dates[0]
        mature_end = session_before(dates, all_dates[-1], MATURITY_SESSIONS)
        result.append({
            "fold_id": f"fold{year}",
            "train_start": train_start,
            "train_end": session_before(dates, test_start, MATURITY_SESSIONS),
            "test_start": test_start,
            "test_end": mature_end,
            "embargo_rule": "4 official open sessions (3D label settlement)",
        })
    return result


def preflight(dates: list[str]) -> tuple[dict[str, object], dict[str, object]]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy["training_window_hard_constraints"]["mode"] != "expanding_available_history":
        raise RuntimeError("blocked_fail_closed_training_window_policy")
    horizon = next(item for item in HORIZONS if item.key == "3d")
    metadata = json.loads(horizon.metadata_path.read_text(encoding="utf-8"))
    feature_connection = duckdb.connect(str(FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(LABEL_DB), read_only=True)
    try:
        feature_min = feature_connection.execute(
            f"SELECT min(trade_date) FROM {quote(FEATURE_TABLE)} WHERE trade_date <= ?", [DEVELOPMENT_END]
        ).fetchone()[0]
        label_min = label_connection.execute(
            f"SELECT min(trade_date) FROM {quote(LABEL_TABLE)} WHERE trade_date <= ?", [DEVELOPMENT_END]
        ).fetchone()[0]
        train_start = max(str(feature_min), str(label_min))
        available = {str(row[1]) for row in feature_connection.execute(f"PRAGMA table_info({quote(FEATURE_TABLE)})").fetchall()}
        requested = [str(value) for value in metadata["feature_columns"]]
        aliases = resolve_savedmodel_feature_aliases(requested, available)
        if aliases["missing"]:
            raise RuntimeError(f"blocked_fail_closed_3d_feature_aliases:{aliases['missing']}")
        alias_map = dict(aliases["alias_pairs"])
        resolved = [alias_map.get(value, value) for value in requested]
        if len(resolved) != 15 or len(set(resolved)) != len(resolved):
            raise RuntimeError("blocked_fail_closed_3d_feature_contract")
        for column in ("executable_1d_open_return", "executable_3d_open_return"):
            count = label_connection.execute(
                f"SELECT count({quote(column)}) FROM {quote(LABEL_TABLE)} WHERE trade_date BETWEEN ? AND ?",
                [train_start, DEVELOPMENT_END],
            ).fetchone()[0]
            if not count:
                raise RuntimeError(f"blocked_fail_closed_missing_label:{column}")
    finally:
        feature_connection.close()
        label_connection.close()
    return {
        "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
        "read_windows": {"features_and_labels": [train_start, DEVELOPMENT_END], "2025": "not_read", "2026_plus": "not_read"},
        "feature_db_sha256": sha256_file(FEATURE_DB),
        "label_db_sha256": sha256_file(LABEL_DB),
        "official_calendar_sha256": sha256_file(CALENDAR_2021_2024),
        "policy_sha256": sha256_file(POLICY_PATH),
        "production_metadata_sha256": sha256_file(horizon.metadata_path),
        "feature_alias_pairs": aliases["alias_pairs"],
        "folds": folds(dates, train_start),
        "production_unchanged": True,
    }, {"metadata": metadata, "features": resolved, "folds": folds(dates, train_start)}


def combined_target(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["target_1d", "target_3d"]).copy()
    valid["rank_1d"] = valid.groupby("trade_date", sort=False)["target_1d"].rank(method="average", pct=True)
    valid["rank_3d"] = valid.groupby("trade_date", sort=False)["target_3d"].rank(method="average", pct=True)
    # The lower rank is the consensus score: both horizons must agree.
    valid["candidate_training_target"] = np.minimum(valid["rank_1d"], valid["rank_3d"])
    if not np.isfinite(valid["candidate_training_target"]).all():
        raise RuntimeError("blocked_fail_closed_candidate_target_nonfinite")
    return valid


def load_baseline(fold: dict[str, str]) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        root = BASELINE_ROOT.as_posix()
        baseline = connection.execute(
            f"""
            SELECT trade_date, stock_code, pred_prob AS baseline_pred_prob
            FROM read_parquet('{root}/3d_oof.parquet')
            WHERE fold_id = ? AND label_mature_within_dev
              AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date, stock_code
            """,
            [fold["fold_id"], fold["test_start"], fold["test_end"]],
        ).fetchdf()
    finally:
        connection.close()
    return baseline


def compare_metrics(baseline: dict[str, object], candidate: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if candidate["rank_ic_mean"] < baseline["rank_ic_mean"]:
        failures.append("rank_ic_not_weaker")
    for count in (1, 3, 5, 10):
        key = f"top{count}_excess_mean"
        if candidate[key] < baseline[key]:
            failures.append(f"top{count}_not_weaker")
    if candidate["top10_turnover_proxy"] > baseline["top10_turnover_proxy"]:
        failures.append("top10_turnover_not_higher")
    return failures


def run(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    dates = strict_open_dates()
    evidence, config = preflight(dates)
    contract = {
        "candidate_id": CANDIDATE_ID,
        "candidate_count": 1,
        "status_before_run": "frozen_single_candidate",
        "development_window": [DEVELOPMENT_START, DEVELOPMENT_END],
        "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
        "training_target": "min(daily_percentile_rank(executable_3d_open_return), daily_percentile_rank(executable_1d_open_return))",
        "model_semantics": "approved 3D feature set and fixed production hyperparameters; production 3D top-quantile weights remain based on original 3D target",
        "evaluation_target": "executable_3d_open_return",
        "free_parameters": 0,
        "hard_gates": ["same_key_finite_no_bj", "aggregate_and_each_fold_rank_ic_not_weaker", "aggregate_and_each_fold_top1_3_5_10_not_weaker", "aggregate_and_each_fold_top10_turnover_not_higher", "deterministic_prediction_replay_3of3"],
        "production_unchanged": True,
        "allow_next_layer_continue": False,
    }
    dump_json(output / "training_contract.json", contract)
    dump_json(output / "preflight.json", evidence)
    feature_connection = duckdb.connect(str(FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(LABEL_DB), read_only=True)
    rows: list[pd.DataFrame] = []
    fold_results: dict[str, object] = {}
    try:
        for fold in config["folds"]:
            train_features = query_features(feature_connection, config["features"], fold["train_start"], fold["train_end"])
            train_1d = query_labels(label_connection, "executable_1d_open_return", fold["train_start"], fold["train_end"]).rename(columns={"target": "target_1d"})
            train_3d = query_labels(label_connection, "executable_3d_open_return", fold["train_start"], fold["train_end"]).rename(columns={"target": "target_3d"})
            train = train_features.merge(train_1d, on=["trade_date", "stock_code"], how="inner", validate="one_to_one").merge(train_3d, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
            candidate_train = combined_target(train)
            original_weight_frame = candidate_train[["trade_date", "stock_code", "target_3d"]].rename(columns={"target_3d": "target"})
            weights = daily_weights(original_weight_frame, config["metadata"].get("sample_weight_config"))
            if len(weights) != len(candidate_train):
                raise RuntimeError("blocked_fail_closed_weight_alignment")
            test_features = query_features(feature_connection, config["features"], fold["test_start"], fold["test_end"])
            test_3d = query_labels(label_connection, "executable_3d_open_return", fold["test_start"], fold["test_end"])
            test = test_features.merge(test_3d, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
            model = xgb.XGBRegressor(**model_params(config["metadata"]))
            matrix_train = candidate_train[config["features"]].astype("float32")
            matrix_test = test[config["features"]].astype("float32")
            model.fit(matrix_train, candidate_train["candidate_training_target"].astype("float32"), sample_weight=weights, verbose=False)
            prediction_1 = model.predict(matrix_test).astype("float64")
            prediction_2 = model.predict(matrix_test).astype("float64")
            prediction_3 = model.predict(matrix_test).astype("float64")
            if not (np.array_equal(prediction_1, prediction_2) and np.array_equal(prediction_1, prediction_3)):
                raise RuntimeError(f"blocked_fail_closed_deterministic_replay:{fold['fold_id']}")
            candidate = test[["trade_date", "stock_code", "target"]].copy()
            candidate["candidate_pred_prob"] = prediction_1
            baseline = load_baseline(fold)
            emitted = candidate.merge(baseline, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
            if len(emitted) != len(candidate) or emitted.duplicated(["trade_date", "stock_code"]).any():
                raise RuntimeError(f"blocked_fail_closed_same_key_baseline:{fold['fold_id']}")
            if emitted.stock_code.astype(str).str.endswith(".BJ").any() or not np.isfinite(emitted[["target", "candidate_pred_prob", "baseline_pred_prob"]].to_numpy(dtype="float64")).all():
                raise RuntimeError(f"blocked_fail_closed_quality:{fold['fold_id']}")
            base_eval, _ = evaluate(emitted.rename(columns={"baseline_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
            candidate_eval, _ = evaluate(emitted.rename(columns={"candidate_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
            fold_results[fold["fold_id"]] = {
                "fold": fold,
                "train_rows": int(len(candidate_train)),
                "test_rows": int(len(emitted)),
                "baseline_metrics": base_eval,
                "candidate_metrics": candidate_eval,
                "failed_gates": compare_metrics(base_eval, candidate_eval),
                "candidate_prediction_sha256": canonical_frame_hash(emitted.rename(columns={"candidate_pred_prob": "pred_prob"}), ["trade_date", "stock_code", "pred_prob"]),
            }
            emitted["fold_id"] = fold["fold_id"]
            rows.append(emitted)
    finally:
        feature_connection.close()
        label_connection.close()
    oof = pd.concat(rows, ignore_index=True)
    if oof.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("blocked_fail_closed_aggregate_duplicate")
    base_aggregate, _ = evaluate(oof.rename(columns={"baseline_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
    candidate_aggregate, _ = evaluate(oof.rename(columns={"candidate_pred_prob": "pred_prob"})[["trade_date", "stock_code", "target", "pred_prob"]])
    aggregate_failures = compare_metrics(base_aggregate, candidate_aggregate)
    failures = sorted(set(aggregate_failures + [item for result in fold_results.values() for item in result["failed_gates"]]))
    passed = not failures
    oof.to_parquet(output / "baseline_candidate_same_key_oof.parquet", index=False)
    summary = {
        "candidate_id": CANDIDATE_ID,
        "status": "completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search",
        "development_window": [DEVELOPMENT_START, DEVELOPMENT_END],
        "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
        "same_key_rows": int(len(oof)),
        "same_key_duplicate_groups": int(oof.duplicated(["trade_date", "stock_code"]).sum()),
        "bj_rows": int(oof.stock_code.astype(str).str.endswith(".BJ").sum()),
        "null_or_nonfinite_rows": int((~np.isfinite(oof[["target", "baseline_pred_prob", "candidate_pred_prob"]].to_numpy(dtype="float64")).all(axis=1)).sum()),
        "aggregate": {"baseline_metrics": base_aggregate, "candidate_metrics": candidate_aggregate, "failed_gates": aggregate_failures},
        "folds": fold_results,
        "hard_gate_passed": passed,
        "failed_gates": failures,
        "baseline_score_sha256": canonical_frame_hash(oof.rename(columns={"baseline_pred_prob": "pred_prob"}), ["trade_date", "stock_code", "pred_prob"]),
        "candidate_score_sha256": canonical_frame_hash(oof.rename(columns={"candidate_pred_prob": "pred_prob"}), ["trade_date", "stock_code", "pred_prob"]),
        "production_unchanged": True,
        "allow_next_layer_continue": False,
    }
    dump_json(output / "evaluation_summary.json", summary)
    dump_json(output / "research_candidate_manifest.json", {**contract, **summary})
    dump_json(output / "hash_inventory.json", {
        "script_sha256": sha256_file(Path(__file__)),
        "training_contract_sha256": sha256_file(output / "training_contract.json"),
        "preflight_sha256": sha256_file(output / "preflight.json"),
        "evaluation_summary_sha256": sha256_file(output / "evaluation_summary.json"),
        "oof_sha256": sha256_file(output / "baseline_candidate_same_key_oof.parquet"),
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    return run(Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
