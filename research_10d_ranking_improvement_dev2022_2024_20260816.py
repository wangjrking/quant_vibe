"""Research-only, strict pre-2025 10D ranking improvement build.

The build is intentionally low freedom: it compares the frozen production
10D training specification with one predeclared change, a fixed chronological
recency multiplier.  It never queries data after 2024-12-13 and writes only to
the research report directory supplied by --output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

from incremental_formal_l4_duckdb_mainline import resolve_savedmodel_feature_aliases


FEATURE_DB = Path("quant/data_file/production_assets/duckdb/l3_feature_current.duckdb")
LABEL_DB = Path("quant/data_file/production_assets/duckdb/l3_label_current.duckdb")
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
METADATA_PATH = Path("quant/data_file/production_assets/production_models/l4/10d/model_metadata.json")
LABEL_COLUMN = "executable_10d_open_return"
DEVELOPMENT_CLOSED_AFTER = "20241213"
RECENCY_TAIL_DAYS = 252
RECENCY_MULTIPLIER = 1.5
TOP_PCT = 0.005
TOP_MULTIPLIER = 8.0
EMBARGO_DAYS = 10


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


# These dates are intentionally predeclared and no test result selects a model
# or changes them.  The 10-session gap is validated against the actual calendar.
FOLDS = (
    Fold("fold01", "20220104", "20220601", "20220620", "20221230"),
    Fold("fold02", "20220104", "20221230", "20230130", "20231229"),
    Fold("fold03", "20220104", "20231229", "20240129", "20241213"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def date_list(connection: duckdb.DuckDBPyConnection, start: str, end: str) -> list[str]:
    return [
        str(row[0]).replace("-", "")
        for row in connection.execute(
            f"SELECT DISTINCT trade_date FROM {quote(FEATURE_TABLE)} "
            "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
            [start, end],
        ).fetchall()
    ]


def validate_fold_calendar(connection: duckdb.DuckDBPyConnection, fold: Fold) -> dict[str, Any]:
    dates = date_list(connection, fold.train_start, fold.test_end)
    train_end_index = dates.index(fold.train_end)
    test_start_index = dates.index(fold.test_start)
    gap = test_start_index - train_end_index - 1
    if gap < EMBARGO_DAYS:
        raise RuntimeError(f"{fold.fold_id}: embargo breach ({gap} < {EMBARGO_DAYS})")
    return {"trade_days": len(dates), "embargo_trade_days": gap}


def load_metadata() -> dict[str, Any]:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("feature_count") != len(metadata.get("feature_columns", [])):
        raise RuntimeError("production 10D metadata feature count is inconsistent")
    return metadata


def query_fold_frame(
    feature_connection: duckdb.DuckDBPyConnection,
    label_connection: duckdb.DuckDBPyConnection,
    feature_columns: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    # All date predicates are explicit; this is the enforcement point for
    # sealed 2025 and opened 2026 exclusion.
    if end > DEVELOPMENT_CLOSED_AFTER or start > DEVELOPMENT_CLOSED_AFTER:
        raise RuntimeError(f"development boundary violation: {start}..{end}")
    select_features = ", ".join(quote(column) for column in feature_columns)
    feature_frame = feature_connection.execute(
        f"SELECT trade_date, stock_code, {select_features} "
        f"FROM {quote(FEATURE_TABLE)} "
        "WHERE trade_date BETWEEN ? AND ? AND stock_code NOT LIKE '%.BJ' "
        "ORDER BY trade_date, stock_code",
        [start, end],
    ).fetchdf()
    label_frame = label_connection.execute(
        f"SELECT trade_date, stock_code, {quote(LABEL_COLUMN)} AS target "
        f"FROM {quote(LABEL_TABLE)} "
        "WHERE trade_date BETWEEN ? AND ? AND stock_code NOT LIKE '%.BJ' "
        "ORDER BY trade_date, stock_code",
        [start, end],
    ).fetchdf()
    frame = feature_frame.merge(label_frame, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    if frame["trade_date"].max() > DEVELOPMENT_CLOSED_AFTER:
        raise RuntimeError("queried forbidden post-2024 data")
    if frame["stock_code"].str.endswith(".BJ").any():
        raise RuntimeError("no-BJ contract violated")
    if frame.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("duplicate key detected in research input")
    return frame


def production_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(frame), dtype=np.float32)
    valid = frame["target"].notna()
    for _, group in frame.loc[valid].groupby("trade_date", sort=False):
        count = len(group)
        top_count = max(1, int(math.ceil(count * TOP_PCT)))
        top_index = group["target"].nlargest(top_count, keep="first").index
        weights[top_index] = TOP_MULTIPLIER
    return weights


def candidate_weights(frame: pd.DataFrame, train_dates: list[str]) -> np.ndarray:
    weights = production_weights(frame)
    recent_dates = set(train_dates[-RECENCY_TAIL_DAYS:])
    weights[frame["trade_date"].isin(recent_dates).to_numpy()] *= RECENCY_MULTIPLIER
    return weights


def xgb_parameters(metadata: dict[str, Any]) -> dict[str, Any]:
    saved = metadata["xgb_params"]
    params = {
        "objective": "reg:squarederror",
        "colsample_bytree": float(saved["colsample_bytree"]),
        "learning_rate": float(saved["learning_rate"]),
        "max_depth": int(saved["max_depth"]),
        # The production model's persisted best iteration is the realized
        # model capacity.  Use it rather than the early-stopping upper bound
        # because fold-local fitting has no access to test-period feedback.
        "n_estimators": int(metadata["best_iteration"]) + 1,
        "n_jobs": int(saved["n_jobs"]),
        "random_state": int(saved["random_state"]),
        "reg_alpha": float(saved["reg_alpha"]),
        "reg_lambda": float(saved["reg_lambda"]),
        "subsample": float(saved["subsample"]),
        "tree_method": "hist",
        "eval_metric": "rmse",
    }
    return params


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], weights: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    valid_train = train["target"].notna().to_numpy()
    if not valid_train.any():
        raise RuntimeError("training labels are all null")
    model = xgb.XGBRegressor(**params)
    x_train = train.loc[valid_train, features].astype("float32")
    y_train = train.loc[valid_train, "target"].astype("float32")
    model.fit(x_train, y_train, sample_weight=weights[valid_train], verbose=False)
    return model.predict(test[features].astype("float32"))


def safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    value = left.corr(right)
    return None if pd.isna(value) else float(value)


def evaluate(predictions: pd.DataFrame) -> dict[str, Any]:
    daily: list[dict[str, Any]] = []
    top_sets: list[set[str]] = []
    for date, frame in predictions.groupby("trade_date", sort=True):
        valid = frame.dropna(subset=["target", "pred_prob"])
        if len(valid) < 20:
            continue
        rank_ic = safe_corr(valid["pred_prob"].rank(method="average"), valid["target"].rank(method="average"))
        ranked = valid.sort_values(["pred_prob", "stock_code"], ascending=[False, True], kind="mergesort")
        all_mean = float(ranked["target"].mean())
        top10 = ranked.head(min(10, len(ranked)))
        top50 = ranked.head(min(50, len(ranked)))
        top_decile = ranked.head(max(1, int(math.ceil(len(ranked) * 0.10))))
        decile = pd.qcut(ranked["pred_prob"].rank(method="first"), 10, labels=False, duplicates="drop")
        decile_frame = ranked.assign(decile=decile).groupby("decile", observed=True).agg(pred_prob=("pred_prob", "mean"), target=("target", "mean"))
        daily.append(
            {
                "trade_date": str(date),
                "rank_ic": rank_ic,
                "top10_excess": float(top10["target"].mean() - all_mean),
                "top50_excess": float(top50["target"].mean() - all_mean),
                "top_decile_excess": float(top_decile["target"].mean() - all_mean),
                "calibration_decile_corr": safe_corr(decile_frame["pred_prob"], decile_frame["target"]),
            }
        )
        top_sets.append(set(top10["stock_code"]))
    if not daily:
        raise RuntimeError("no evaluable test dates")
    daily_frame = pd.DataFrame(daily)
    overlaps = [len(left & right) / 10.0 for left, right in zip(top_sets, top_sets[1:])]
    return {
        "evaluable_trade_days": int(len(daily_frame)),
        "rank_ic_mean": float(daily_frame["rank_ic"].mean()),
        "rank_ic_positive_day_share": float((daily_frame["rank_ic"] > 0).mean()),
        "top10_excess_mean": float(daily_frame["top10_excess"].mean()),
        "top50_excess_mean": float(daily_frame["top50_excess"].mean()),
        "top_decile_excess_mean": float(daily_frame["top_decile_excess"].mean()),
        "calibration_decile_corr_mean": float(daily_frame["calibration_decile_corr"].mean()),
        "top10_turnover_proxy": None if not overlaps else float(1.0 - np.mean(overlaps)),
        "daily_metrics": daily,
    }


def delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return float(candidate[key] - baseline[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="quant/data_file/reports/model_agent_10d_ranking_improvement_dev2022_2024_20260816",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite research output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    metadata = load_metadata()
    if metadata.get("sample_weight_config") != {
        "mode": "daily_top_quantile",
        "top_pct": TOP_PCT,
        "top_multiplier": TOP_MULTIPLIER,
    }:
        raise RuntimeError("production weight contract unexpectedly differs from frozen baseline")
    if not FEATURE_DB.exists() or not LABEL_DB.exists() or not METADATA_PATH.exists():
        raise RuntimeError("required read-only input asset is missing")

    feature_connection = duckdb.connect(str(FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(LABEL_DB), read_only=True)
    try:
        available = {
            str(row[1])
            for row in feature_connection.execute(f"PRAGMA table_info({quote(FEATURE_TABLE)})").fetchall()
        }
        requested_features = [str(item) for item in metadata["feature_columns"]]
        resolved = resolve_savedmodel_feature_aliases(requested_features, available)
        if resolved["missing"]:
            raise RuntimeError(f"unresolved production feature aliases: {resolved['missing']}")
        actual_features = [resolved_name for _, resolved_name in [
            (feature, next((pair[1] for pair in resolved["alias_pairs"] if pair[0] == feature), feature))
            for feature in requested_features
        ]]
        if len(actual_features) != len(requested_features) or len(set(actual_features)) != len(actual_features):
            raise RuntimeError("feature alias mapping changes model feature dimensionality")

        preflight = {
            "asset_role": "research_only_l4_candidate_build",
            "approval_status": "research_only_not_for_l5",
            "development_closed_after": DEVELOPMENT_CLOSED_AFTER,
            "sealed_periods_not_read": ["2025", "2026+"],
            "production_assets_unchanged": True,
            "method": {
                "baseline": "fold-local production_10d_xgb_specification with frozen daily_top_quantile weights",
                "principal_candidate": "same specification plus a predeclared 1.5x sample-weight multiplier on the most recent 252 training sessions",
                "robustness_controls": [],
                "selection_rule": "no post-hoc tuning; candidate is accepted only if predeclared aggregate gate passes",
            },
            "inputs": {
                "feature_db": str(FEATURE_DB),
                "feature_db_sha256": sha256_file(FEATURE_DB),
                "feature_table": FEATURE_TABLE,
                "label_db": str(LABEL_DB),
                "label_db_sha256": sha256_file(LABEL_DB),
                "label_table": LABEL_TABLE,
                "label_column": LABEL_COLUMN,
                "production_metadata": str(METADATA_PATH),
                "production_metadata_sha256": sha256_file(METADATA_PATH),
                "requested_feature_columns": requested_features,
                "resolved_feature_columns": actual_features,
                "qfq_alias_pairs": resolved["alias_pairs"],
            },
            "folds": [fold.__dict__ | validate_fold_calendar(feature_connection, fold) for fold in FOLDS],
        }
        json_dump(output_dir / "preflight.json", preflight)

        params = xgb_parameters(metadata)
        fold_results: list[dict[str, Any]] = []
        all_predictions: list[pd.DataFrame] = []
        for fold in FOLDS:
            train = query_fold_frame(feature_connection, label_connection, actual_features, fold.train_start, fold.train_end)
            test = query_fold_frame(feature_connection, label_connection, actual_features, fold.test_start, fold.test_end)
            train_dates = sorted(train["trade_date"].unique().tolist())
            if train["target"].notna().sum() == 0 or test["target"].notna().sum() == 0:
                raise RuntimeError(f"{fold.fold_id}: labels unavailable in strict development period")
            baseline_pred = fit_predict(train, test, actual_features, production_weights(train), params)
            candidate_pred = fit_predict(train, test, actual_features, candidate_weights(train, train_dates), params)
            baseline_frame = test[["trade_date", "stock_code", "target"]].copy()
            baseline_frame["pred_prob"] = baseline_pred.astype("float64")
            candidate_frame = test[["trade_date", "stock_code", "target"]].copy()
            candidate_frame["pred_prob"] = candidate_pred.astype("float64")
            baseline_metrics = evaluate(baseline_frame)
            candidate_metrics = evaluate(candidate_frame)
            fold_results.append(
                {
                    "fold": fold.__dict__,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "delta": {
                        "rank_ic_mean": delta(candidate_metrics, baseline_metrics, "rank_ic_mean"),
                        "top_decile_excess_mean": delta(candidate_metrics, baseline_metrics, "top_decile_excess_mean"),
                        "top10_excess_mean": delta(candidate_metrics, baseline_metrics, "top10_excess_mean"),
                        "calibration_decile_corr_mean": delta(candidate_metrics, baseline_metrics, "calibration_decile_corr_mean"),
                        "top10_turnover_proxy": delta(candidate_metrics, baseline_metrics, "top10_turnover_proxy"),
                    },
                }
            )
            baseline_frame["variant"] = "baseline"
            candidate_frame["variant"] = "candidate_recency_weight_1p5_tail252"
            baseline_frame["fold_id"] = fold.fold_id
            candidate_frame["fold_id"] = fold.fold_id
            all_predictions.extend([baseline_frame, candidate_frame])
            json_dump(output_dir / f"{fold.fold_id}_summary.json", fold_results[-1])

        summary_frame = pd.DataFrame(fold_results)
        aggregate = {
            metric: float(np.mean([item["delta"][metric] for item in fold_results]))
            for metric in fold_results[0]["delta"]
        }
        pass_gate = (
            aggregate["rank_ic_mean"] > 0
            and aggregate["top_decile_excess_mean"] > 0
            and aggregate["top10_excess_mean"] > 0
            and sum(item["delta"]["rank_ic_mean"] > 0 for item in fold_results) >= 2
            and sum(item["delta"]["top_decile_excess_mean"] > 0 for item in fold_results) >= 2
        )
        result = {
            "asset_role": "research_only_l4_candidate_evaluation",
            "approval_status": "research_only_not_for_l5",
            "development_closed_after": DEVELOPMENT_CLOSED_AFTER,
            "fold_results": fold_results,
            "aggregate_delta_candidate_minus_baseline": aggregate,
            "predeclared_acceptance_gate": {
                "mean_rank_ic_delta_gt_zero": True,
                "mean_top_decile_excess_delta_gt_zero": True,
                "mean_top10_excess_delta_gt_zero": True,
                "at_least_two_of_three_folds_rank_ic_positive": True,
                "at_least_two_of_three_folds_top_decile_positive": True,
                "passed": pass_gate,
            },
            "decision": "freeze_unique_research_candidate_for_strategy_development_ab" if pass_gate else "reject_no_further_search",
            "ready_for_audit_review": True,
            "allow_next_layer_continue": False,
            "production_unchanged": True,
        }
        json_dump(output_dir / "evaluation_summary.json", result)
        pd.concat(all_predictions, ignore_index=True).to_parquet(output_dir / "oof_predictions.parquet", index=False)
        json_dump(
            output_dir / "hash_inventory.json",
            {
                "preflight_sha256": sha256_file(output_dir / "preflight.json"),
                "evaluation_summary_sha256": sha256_file(output_dir / "evaluation_summary.json"),
                "oof_predictions_sha256": sha256_file(output_dir / "oof_predictions.parquet"),
                "script_sha256": sha256_file(Path(__file__)),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0
    finally:
        feature_connection.close()
        label_connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
