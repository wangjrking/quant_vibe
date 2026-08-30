"""Build strict pre-2026 expanding-window PIT/OOF L4 baselines.

This is intentionally a research-only baseline builder.  It recreates the
four current production model specifications with expanding training windows,
but never writes a formal prediction asset or reads data after 2025-12-31.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

from incremental_formal_l4_duckdb_mainline import resolve_savedmodel_feature_aliases


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
FEATURE_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
LABEL_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_label_current.duckdb"
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
POLICY_PATH = ROOT / "quant" / "main" / "config" / "model_training_window_policy_v1_20260829.json"
CALENDAR_2021_2024 = (
    DATA_DIR
    / "runtime"
    / "agent_workspaces"
    / "data-ingestion-agent"
    / "work"
    / "v260_pit_oof_excess_logit_10d_v1_official_trade_cal_20210101_20241231_r1"
    / "official_trade_cal.duckdb"
)
CALENDAR_2025 = (
    DATA_DIR
    / "runtime"
    / "agent_workspaces"
    / "data-ingestion-agent"
    / "work"
    / "v260_rolling252_official_calendar_binding_20250101_20260104_r1"
    / "official_trade_calendar_binding.duckdb"
)
DEVELOPMENT_START = "20220101"
DEVELOPMENT_END = "20251231"
OUTPUT_DEFAULT = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829"


@dataclass(frozen=True)
class Horizon:
    key: str
    label: str
    maturity_sessions: int
    metadata_path: Path


HORIZONS = (
    Horizon("1d", "executable_1d_open_return", 2, DATA_DIR / "production_assets" / "production_models" / "l4" / "1d" / "model_metadata.json"),
    Horizon("3d", "executable_3d_open_return", 4, DATA_DIR / "production_assets" / "production_models" / "l4" / "3d" / "model_metadata.json"),
    Horizon("5d", "executable_5d_open_return", 6, DATA_DIR / "production_assets" / "production_models" / "l4" / "5d" / "model_metadata.json"),
    Horizon("10d", "executable_10d_open_return", 12, DATA_DIR / "production_assets" / "production_models" / "l4" / "10d" / "model_metadata.json"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def canonical_frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.sort_values(["trade_date", "stock_code"], kind="mergesort")[columns]
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, ".17g"))
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        digest.update(("|".join(values) + "\n").encode("utf-8"))
    return digest.hexdigest()


def read_open_dates() -> list[str]:
    dates: list[str] = []
    for path, table in ((CALENDAR_2021_2024, "official_trade_cal"), (CALENDAR_2025, "official_trade_calendar_binding")):
        if not path.exists():
            raise RuntimeError(f"official calendar missing: {path}")
        connection = duckdb.connect(str(path), read_only=True)
        try:
            dates.extend(
                str(row[0])
                for row in connection.execute(
                    f"SELECT cal_date FROM {quote(table)} "
                    "WHERE is_open = 1 AND cal_date <= ? ORDER BY cal_date",
                    [DEVELOPMENT_END],
                ).fetchall()
            )
        finally:
            connection.close()
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise RuntimeError("official calendar is not a unique sorted pre-2026 sequence")
    if dates[0] > "20210104" or dates[-1] != DEVELOPMENT_END:
        raise RuntimeError("official calendar does not close the required 2021-2025 pre-2026 range")
    return dates


def session_before(open_dates: list[str], date: str, sessions: int) -> str:
    index = open_dates.index(date)
    if index < sessions:
        raise RuntimeError(f"insufficient official-calendar history before {date}")
    return open_dates[index - sessions]


def yearly_folds(open_dates: list[str], horizon: Horizon, train_start: str) -> list[dict[str, str]]:
    folds: list[dict[str, str]] = []
    for year in ("2022", "2023", "2024", "2025"):
        test_dates = [date for date in open_dates if date.startswith(year)]
        if not test_dates:
            raise RuntimeError(f"official calendar has no open dates for {year}")
        test_start, test_end = test_dates[0], test_dates[-1]
        train_end = session_before(open_dates, test_start, horizon.maturity_sessions)
        label_cutoff = session_before(open_dates, "20260101" if False else test_end, 0)
        # Evaluation labels are admitted only when their full execution horizon
        # settles inside the frozen development period.
        mature_dates = [date for date in test_dates if open_dates.index(date) + horizon.maturity_sessions < len(open_dates) and open_dates[open_dates.index(date) + horizon.maturity_sessions] <= DEVELOPMENT_END]
        if not mature_dates:
            raise RuntimeError(f"{horizon.key}/{year}: no test dates mature inside development")
        folds.append({
            "fold_id": f"fold{year}",
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "mature_label_cutoff": mature_dates[-1],
            "embargo_rule": f"{horizon.maturity_sessions} official open sessions (label settlement)",
            "maturity_sessions": str(horizon.maturity_sessions),
        })
    return folds


def safe_mean(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return None if not finite else float(np.mean(finite))


def daily_weights(frame: pd.DataFrame, config: dict[str, Any] | None) -> np.ndarray | None:
    if config is None:
        return None
    if config != {"mode": "daily_top_quantile", "top_pct": config["top_pct"], "top_multiplier": config["top_multiplier"]}:
        raise RuntimeError("unsupported production sample weight configuration")
    # The caller may have removed null-label rows.  Resetting here makes the
    # returned NumPy positions match the training matrix, not source row ids.
    frame = frame.reset_index(drop=True)
    weights = np.ones(len(frame), dtype=np.float32)
    valid = frame["target"].notna()
    for _, group in frame.loc[valid].groupby("trade_date", sort=False):
        count = len(group)
        top_count = max(1, int(math.floor(count * float(config["top_pct"]))))
        winners = group.nlargest(top_count, "target", keep="first").index
        weights[winners] = float(config["top_multiplier"])
    return weights


def model_params(metadata: dict[str, Any]) -> dict[str, Any]:
    saved = metadata["xgb_params"]
    trees = int(metadata.get("best_iteration", int(saved["n_estimators"]) - 1)) + 1
    return {
        "objective": "reg:squarederror",
        "colsample_bytree": float(saved["colsample_bytree"]),
        "learning_rate": float(saved["learning_rate"]),
        "max_depth": int(saved["max_depth"]),
        "n_estimators": trees,
        "n_jobs": int(saved["n_jobs"]),
        "random_state": int(saved["random_state"]),
        "reg_alpha": float(saved["reg_alpha"]),
        "reg_lambda": float(saved["reg_lambda"]),
        "subsample": float(saved["subsample"]),
        "tree_method": "hist",
        "device": str(saved.get("device") or "cpu"),
        "eval_metric": "rmse",
    }


def query_features(connection: duckdb.DuckDBPyConnection, features: list[str], start: str, end: str) -> pd.DataFrame:
    columns = ", ".join(quote(name) for name in features)
    return connection.execute(
        f"SELECT trade_date, stock_code, {columns} FROM {quote(FEATURE_TABLE)} "
        "WHERE trade_date BETWEEN ? AND ? AND stock_code NOT LIKE '%.BJ' "
        "ORDER BY trade_date, stock_code",
        [start, end],
    ).fetchdf()


def query_labels(connection: duckdb.DuckDBPyConnection, label: str, start: str, end: str) -> pd.DataFrame:
    return connection.execute(
        f"SELECT trade_date, stock_code, {quote(label)} AS target FROM {quote(LABEL_TABLE)} "
        "WHERE trade_date BETWEEN ? AND ? AND stock_code NOT LIKE '%.BJ' "
        "ORDER BY trade_date, stock_code",
        [start, end],
    ).fetchdf()


def evaluate(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    daily: list[dict[str, Any]] = []
    top_sets: list[set[str]] = []
    for date, group in frame.dropna(subset=["target"]).groupby("trade_date", sort=True):
        ranked = group.sort_values(["pred_prob", "stock_code"], ascending=[False, True], kind="mergesort")
        target_rank = ranked["target"].rank(method="average")
        score_rank = ranked["pred_prob"].rank(method="average")
        all_mean = float(ranked["target"].mean())
        record: dict[str, Any] = {"trade_date": str(date), "rank_ic": float(score_rank.corr(target_rank))}
        for count in (1, 3, 5, 10, 20):
            record[f"top{count}_excess"] = float(ranked.head(count)["target"].mean() - all_mean)
        daily.append(record)
        top_sets.append(set(ranked.head(10)["stock_code"]))
    if not daily:
        raise RuntimeError("no mature OOF dates are evaluable")
    daily_frame = pd.DataFrame(daily)
    overlap = [len(left & right) / 10.0 for left, right in zip(top_sets, top_sets[1:])]
    result: dict[str, Any] = {
        "evaluable_trade_days": int(len(daily_frame)),
        "rank_ic_mean": float(daily_frame["rank_ic"].mean()),
        "rank_ic_positive_day_share": float((daily_frame["rank_ic"] > 0).mean()),
        "top10_turnover_proxy": None if not overlap else float(1.0 - np.mean(overlap)),
    }
    for count in (1, 3, 5, 10, 20):
        result[f"top{count}_excess_mean"] = float(daily_frame[f"top{count}_excess"].mean())
    for window in (20, 63, 126):
        tail = daily_frame.tail(window)
        result[f"recent{window}"] = {
            "trade_days": int(len(tail)), "rank_ic_mean": float(tail["rank_ic"].mean()),
            "top10_excess_mean": float(tail["top10_excess"].mean()),
        }
    daily_frame["year"] = daily_frame["trade_date"].str.slice(0, 4)
    daily_frame["month"] = daily_frame["trade_date"].str.slice(0, 6)
    result["annual_stability"] = [
        {"year": year, "trade_days": int(len(group)), "rank_ic_mean": float(group["rank_ic"].mean()), "top10_excess_mean": float(group["top10_excess"].mean())}
        for year, group in daily_frame.groupby("year", sort=True)
    ]
    result["monthly_stability"] = [
        {"month": month, "trade_days": int(len(group)), "rank_ic_mean": float(group["rank_ic"].mean()), "top10_excess_mean": float(group["top10_excess"].mean())}
        for month, group in daily_frame.groupby("month", sort=True)
    ]
    return result, daily_frame


def preflight(open_dates: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    constraints = policy["training_window_hard_constraints"]
    if constraints["mode"] != "expanding_available_history" or constraints["fixed_four_year_training_cutoff_forbidden"] is not True:
        raise RuntimeError("expanding training-window policy is not active")
    feature_connection = duckdb.connect(str(FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(LABEL_DB), read_only=True)
    try:
        feature_range = feature_connection.execute(
            f"SELECT min(trade_date), max(trade_date), count(*) FROM {quote(FEATURE_TABLE)} WHERE trade_date <= ?",
            [DEVELOPMENT_END],
        ).fetchone()
        label_range = label_connection.execute(
            f"SELECT min(trade_date), max(trade_date), count(*) FROM {quote(LABEL_TABLE)} WHERE trade_date <= ?",
            [DEVELOPMENT_END],
        ).fetchone()
        train_start = max(str(feature_range[0]), str(label_range[0]))
        if train_start > "20100104":
            raise RuntimeError(f"joint PIT history starts too late: {train_start}")
        available = {str(row[1]) for row in feature_connection.execute(f"PRAGMA table_info({quote(FEATURE_TABLE)})").fetchall()}
        prepared: dict[str, Any] = {}
        for horizon in HORIZONS:
            metadata = json.loads(horizon.metadata_path.read_text(encoding="utf-8"))
            requested = [str(value) for value in metadata["feature_columns"]]
            aliases = resolve_savedmodel_feature_aliases(requested, available)
            if aliases["missing"]:
                raise RuntimeError(f"{horizon.key}: unresolved approved feature aliases: {aliases['missing']}")
            alias_lookup = dict(aliases["alias_pairs"])
            resolved = [alias_lookup.get(value, value) for value in requested]
            if len(resolved) != len(requested) or len(set(resolved)) != len(resolved):
                raise RuntimeError(f"{horizon.key}: feature contract dimension changed")
            label_count = label_connection.execute(
                f"SELECT count({quote(horizon.label)}) FROM {quote(LABEL_TABLE)} WHERE trade_date BETWEEN ? AND ?",
                [train_start, DEVELOPMENT_END],
            ).fetchone()[0]
            if not label_count:
                raise RuntimeError(f"{horizon.key}: no pre-2026 labels available")
            prepared[horizon.key] = {"metadata": metadata, "requested": requested, "resolved": resolved, "aliases": aliases, "folds": yearly_folds(open_dates, horizon, train_start)}
        evidence = {
            "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
            "development_closed_after": DEVELOPMENT_END,
            "forbidden_periods": ["2026+"],
            "policy_sha256": sha256_file(POLICY_PATH),
            "feature_db_sha256": sha256_file(FEATURE_DB),
            "label_db_sha256": sha256_file(LABEL_DB),
            "official_calendar_2021_2024_sha256": sha256_file(CALENDAR_2021_2024),
            "official_calendar_2025_sha256": sha256_file(CALENDAR_2025),
            "official_open_dates": {"count": len(open_dates), "first": open_dates[0], "last": open_dates[-1]},
            "joint_pit_train_start": train_start,
            "feature_range_pre2026": feature_range,
            "label_range_pre2026": label_range,
            "horizons": {
                key: {
                    "feature_count": len(value["resolved"]),
                    "maturity_sessions": next(item.maturity_sessions for item in HORIZONS if item.key == key),
                    "folds": value["folds"],
                }
                for key, value in prepared.items()
            },
            "production_unchanged": True,
        }
        return evidence, prepared
    finally:
        feature_connection.close()
        label_connection.close()


def run(output_dir: Path, preflight_only: bool) -> int:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing research root: {output_dir}")
    output_dir.mkdir(parents=True)
    open_dates = read_open_dates()
    evidence, prepared = preflight(open_dates)
    dump_json(output_dir / "preflight.json", evidence)
    if preflight_only:
        return 0
    feature_connection = duckdb.connect(str(FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(LABEL_DB), read_only=True)
    all_rows: list[pd.DataFrame] = []
    horizon_summaries: dict[str, Any] = {}
    try:
        for horizon in HORIZONS:
            info = prepared[horizon.key]
            metadata = info["metadata"]
            params = model_params(metadata)
            weight_config = metadata.get("sample_weight_config")
            fold_summaries = []
            horizon_rows: list[pd.DataFrame] = []
            for fold in info["folds"]:
                train_features = query_features(feature_connection, info["resolved"], fold["train_start"], fold["train_end"])
                train_labels = query_labels(label_connection, horizon.label, fold["train_start"], fold["train_end"])
                train = train_features.merge(train_labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
                test_features = query_features(feature_connection, info["resolved"], fold["test_start"], fold["test_end"])
                # Labels after mature_label_cutoff are deliberately not queried.
                test_labels = query_labels(label_connection, horizon.label, fold["test_start"], fold["mature_label_cutoff"])
                test = test_features.merge(test_labels, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
                labeled_train = train.dropna(subset=["target"]).copy()
                if labeled_train.empty:
                    raise RuntimeError(f"{horizon.key}/{fold['fold_id']}: no mature training labels")
                if labeled_train["trade_date"].max() != fold["train_end"]:
                    raise RuntimeError(f"{horizon.key}/{fold['fold_id']}: expanding train_end does not close")
                weights = daily_weights(labeled_train, weight_config)
                model = xgb.XGBRegressor(**params)
                model.fit(labeled_train[info["resolved"]].astype("float32"), labeled_train["target"].astype("float32"), sample_weight=weights, verbose=False)
                pred_one = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
                # Re-score the frozen fitted model twice; all hashes must agree.
                pred_two = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
                pred_three = model.predict(test[info["resolved"]].astype("float32")).astype("float64")
                if not (np.array_equal(pred_one, pred_two) and np.array_equal(pred_one, pred_three)):
                    raise RuntimeError(f"{horizon.key}/{fold['fold_id']}: deterministic prediction replay failed")
                scored = test[["trade_date", "stock_code", "target"]].copy()
                scored["pred_prob"] = pred_one
                scored["fold_id"] = fold["fold_id"]
                scored["horizon"] = horizon.key
                scored["label_mature_within_dev"] = scored["trade_date"] <= fold["mature_label_cutoff"]
                if scored.duplicated(["trade_date", "stock_code"]).any() or scored["stock_code"].str.endswith(".BJ").any():
                    raise RuntimeError(f"{horizon.key}/{fold['fold_id']}: key/no-BJ quality gate failed")
                if scored["pred_prob"].isna().any() or not np.isfinite(scored["pred_prob"]).all():
                    raise RuntimeError(f"{horizon.key}/{fold['fold_id']}: prediction finite gate failed")
                metrics, daily = evaluate(scored.loc[scored["label_mature_within_dev"]].copy())
                fold_summary = {"fold": fold, "train_rows_before_label_filter": int(len(train)), "train_rows": int(len(labeled_train)), "train_label_null_rows_excluded": int(train["target"].isna().sum()), "test_rows": int(len(scored)), "mature_test_rows": int(scored["label_mature_within_dev"].sum()), "metrics": metrics, "prediction_sha256": canonical_frame_hash(scored, ["trade_date", "stock_code", "pred_prob"])}
                dump_json(output_dir / f"{horizon.key}_{fold['fold_id']}_summary.json", fold_summary)
                fold_summaries.append(fold_summary)
                horizon_rows.append(scored)
            horizon_oof = pd.concat(horizon_rows, ignore_index=True)
            if horizon_oof.duplicated(["trade_date", "stock_code"]).any():
                raise RuntimeError(f"{horizon.key}: OOF same-key closure failed")
            mature = horizon_oof.loc[horizon_oof["label_mature_within_dev"]].copy()
            aggregate_metrics, daily = evaluate(mature)
            daily.to_parquet(output_dir / f"{horizon.key}_daily_metrics.parquet", index=False)
            horizon_oof.to_parquet(output_dir / f"{horizon.key}_oof.parquet", index=False)
            horizon_summaries[horizon.key] = {
                "label": horizon.label,
                "production_metadata_path": str(horizon.metadata_path),
                "production_metadata_sha256": sha256_file(horizon.metadata_path),
                "requested_feature_columns": info["requested"],
                "resolved_feature_columns": info["resolved"],
                "qfq_alias_pairs": info["aliases"]["alias_pairs"],
                "params": params,
                "sample_weight_config": weight_config,
                "folds": fold_summaries,
                "aggregate_metrics_mature_only": aggregate_metrics,
                "oof_rows": int(len(horizon_oof)),
                "mature_oof_rows": int(len(mature)),
                "oof_sha256": sha256_file(output_dir / f"{horizon.key}_oof.parquet"),
                "daily_metrics_sha256": sha256_file(output_dir / f"{horizon.key}_daily_metrics.parquet"),
            }
            all_rows.append(horizon_oof)
    finally:
        feature_connection.close()
        label_connection.close()
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_parquet(output_dir / "expanding_pit_oof_baselines.parquet", index=False)
    manifest = {
        "asset_role": "l4_research_strict_pit_oof_baseline",
        "approval_status": "research_only_not_for_l5",
        "training_window_mode": "expanding_available_history",
        "development_period": [DEVELOPMENT_START, DEVELOPMENT_END],
        "validation_2026_closed": True,
        "production_unchanged": True,
        "formal_manifest_unchanged": True,
        "horizons": horizon_summaries,
        "combined_oof_path": str(output_dir / "expanding_pit_oof_baselines.parquet"),
        "combined_oof_sha256": sha256_file(output_dir / "expanding_pit_oof_baselines.parquet"),
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    dump_json(output_dir / "research_baseline_manifest.json", manifest)
    handoff = {
        "status": "completed_ready_for_audit",
        "asset_role": "research_only_l4_baseline_handoff",
        "manifest": str(output_dir / "research_baseline_manifest.json"),
        "manifest_sha256": sha256_file(output_dir / "research_baseline_manifest.json"),
        "strict_pit_oof": True,
        "expanding_available_history": True,
        "validation_2026_closed": True,
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "production_unchanged": True,
    }
    dump_json(output_dir / "audit_handoff.json", handoff)
    dump_json(output_dir / "hash_inventory.json", {
        "script_sha256": sha256_file(Path(__file__)),
        "preflight_sha256": sha256_file(output_dir / "preflight.json"),
        "manifest_sha256": sha256_file(output_dir / "research_baseline_manifest.json"),
        "handoff_sha256": sha256_file(output_dir / "audit_handoff.json"),
        "combined_oof_sha256": sha256_file(output_dir / "expanding_pit_oof_baselines.parquet"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    return run(args.output_dir, args.preflight_only)


if __name__ == "__main__":
    raise SystemExit(main())
