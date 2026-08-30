"""Append embargo-only score-smoothing warmup to a frozen research OOF asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

import research_10d_rank_target_dev2022_2024_20260816 as rank_target
import research_10d_ranking_improvement_dev2022_2024_20260816 as base


OUTPUT_DEFAULT = "quant/data_file/reports/model_agent_10d_rank_target_warmup_oof_dev2022_2024_20260816"
BASELINE_OOF = Path("quant/data_file/reports/model_agent_10d_ranking_improvement_dev2022_2024_20260816/oof_predictions.parquet")
CANDIDATE_OOF = Path("quant/data_file/reports/model_agent_10d_rank_target_dev2022_2024_20260816/rank_target_candidate_oof_predictions.parquet")
CANDIDATE_MANIFEST = Path("quant/data_file/reports/model_agent_10d_rank_target_dev2022_2024_20260816/research_candidate_manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_feature_frame(
    connection: duckdb.DuckDBPyConnection,
    features: list[str],
    dates: list[str],
) -> pd.DataFrame:
    if not dates or max(dates) > base.DEVELOPMENT_CLOSED_AFTER:
        raise RuntimeError("warmup request crosses sealed development boundary")
    placeholders = ", ".join("?" for _ in dates)
    columns = ", ".join(base.quote(column) for column in features)
    frame = connection.execute(
        f"SELECT trade_date, stock_code, {columns} FROM {base.quote(base.FEATURE_TABLE)} "
        f"WHERE trade_date IN ({placeholders}) AND stock_code NOT LIKE '%.BJ' "
        "ORDER BY trade_date, stock_code",
        dates,
    ).fetchdf()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    if len(frame) == 0 or frame.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("warmup feature key domain is invalid")
    if frame["stock_code"].str.endswith(".BJ").any():
        raise RuntimeError("warmup no-BJ contract failed")
    return frame


def fit_model(train: pd.DataFrame, features: list[str], target: pd.Series, weights: np.ndarray, params: dict[str, Any]) -> xgb.XGBRegressor:
    valid = target.notna().to_numpy()
    if not valid.any():
        raise RuntimeError("warmup model train label is empty")
    model = xgb.XGBRegressor(**params)
    model.fit(
        train.loc[valid, features].astype("float32"),
        target.loc[valid].astype("float32"),
        sample_weight=weights[valid],
        verbose=False,
    )
    return model


def source_rows(path: Path, variant: str, expected_folds: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["variant"] == variant, ["fold_id", "trade_date", "stock_code", "pred_prob"]].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    if frame.empty or set(frame["fold_id"].unique()) != expected_folds:
        raise RuntimeError(f"source OOF fold domain mismatch: {path}")
    if frame["trade_date"].max() > base.DEVELOPMENT_CLOSED_AFTER or frame.duplicated(["fold_id", "trade_date", "stock_code"]).any():
        raise RuntimeError(f"source OOF quality failed: {path}")
    return frame


def same_prediction(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(left.astype("float64"), right.astype("float64"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite warmup research directory: {output_dir}")
    if not (BASELINE_OOF.exists() and CANDIDATE_OOF.exists() and CANDIDATE_MANIFEST.exists()):
        raise RuntimeError("frozen OOF input or candidate manifest is missing")
    candidate_manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    if not candidate_manifest.get("frozen") or candidate_manifest.get("approved_for_l5"):
        raise RuntimeError("candidate manifest is not a frozen research-only asset")
    output_dir.mkdir(parents=True)

    expected_folds = {fold.fold_id for fold in base.FOLDS}
    baseline_test = source_rows(BASELINE_OOF, "baseline", expected_folds)
    candidate_test = source_rows(CANDIDATE_OOF, "rank_target_candidate", expected_folds)
    source_hashes_before = {
        "baseline_oof_sha256": sha256_file(BASELINE_OOF),
        "candidate_oof_sha256": sha256_file(CANDIDATE_OOF),
        "candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST),
    }
    metadata = base.load_metadata()
    feature_connection = duckdb.connect(str(base.FEATURE_DB), read_only=True)
    label_connection = duckdb.connect(str(base.LABEL_DB), read_only=True)
    try:
        available = {str(row[1]) for row in feature_connection.execute(f"PRAGMA table_info({base.quote(base.FEATURE_TABLE)})").fetchall()}
        aliases = base.resolve_savedmodel_feature_aliases([str(item) for item in metadata["feature_columns"]], available)
        if aliases["missing"]:
            raise RuntimeError(f"production feature aliases drifted: {aliases['missing']}")
        alias_lookup = dict(aliases["alias_pairs"])
        features = [alias_lookup.get(str(item), str(item)) for item in metadata["feature_columns"]]
        params = base.xgb_parameters(metadata)
        warmup_rows: list[pd.DataFrame] = []
        fold_details: list[dict[str, Any]] = []
        for fold in base.FOLDS:
            full_dates = base.date_list(feature_connection, fold.train_end, fold.test_start)
            warmup_dates = [date for date in full_dates if fold.train_end < date < fold.test_start]
            if len(warmup_dates) < 7:
                raise RuntimeError(f"{fold.fold_id}: fewer than seven embargo warmup sessions")
            warmup = load_feature_frame(feature_connection, features, warmup_dates)
            test_features = load_feature_frame(feature_connection, features, base.date_list(feature_connection, fold.test_start, fold.test_end))
            train = base.query_fold_frame(feature_connection, label_connection, features, fold.train_start, fold.train_end)
            weights = base.production_weights(train)
            baseline_model = fit_model(train, features, train["target"], weights, params)
            candidate_model = fit_model(train, features, rank_target.rank_target(train), weights, params)
            replay_baseline = baseline_model.predict(test_features[features].astype("float32")).astype("float64")
            replay_candidate = candidate_model.predict(test_features[features].astype("float32")).astype("float64")
            expected_baseline = baseline_test.loc[baseline_test["fold_id"] == fold.fold_id].sort_values(["trade_date", "stock_code"], kind="mergesort")
            expected_candidate = candidate_test.loc[candidate_test["fold_id"] == fold.fold_id].sort_values(["trade_date", "stock_code"], kind="mergesort")
            test_features = test_features.sort_values(["trade_date", "stock_code"], kind="mergesort")
            if not test_features[["trade_date", "stock_code"]].reset_index(drop=True).equals(expected_baseline[["trade_date", "stock_code"]].reset_index(drop=True)):
                raise RuntimeError(f"{fold.fold_id}: baseline OOF test keys differ from frozen source")
            if not test_features[["trade_date", "stock_code"]].reset_index(drop=True).equals(expected_candidate[["trade_date", "stock_code"]].reset_index(drop=True)):
                raise RuntimeError(f"{fold.fold_id}: candidate OOF test keys differ from frozen source")
            if not same_prediction(replay_baseline, expected_baseline["pred_prob"].to_numpy()):
                raise RuntimeError(f"{fold.fold_id}: baseline model replay differs from frozen OOF")
            if not same_prediction(replay_candidate, expected_candidate["pred_prob"].to_numpy()):
                raise RuntimeError(f"{fold.fold_id}: candidate model replay differs from frozen OOF")
            warmup = warmup.sort_values(["trade_date", "stock_code"], kind="mergesort")
            for variant, model in (("baseline", baseline_model), ("rank_target_candidate", candidate_model)):
                frame = warmup[["trade_date", "stock_code"]].copy()
                frame["pred_prob"] = model.predict(warmup[features].astype("float32")).astype("float64")
                frame["fold_id"] = fold.fold_id
                frame["variant"] = variant
                frame["row_role"] = "warmup"
                warmup_rows.append(frame)
            fold_details.append({
                "fold_id": fold.fold_id,
                "train_end": fold.train_end,
                "warmup_start": warmup_dates[0],
                "warmup_end": warmup_dates[-1],
                "test_start": fold.test_start,
                "warmup_trade_days": len(warmup_dates),
                "warmup_rows_per_variant": int(len(warmup)),
                "warmup_strictly_after_train_and_before_test": True,
                "baseline_test_replay_exact": True,
                "candidate_test_replay_exact": True,
            })
    finally:
        feature_connection.close()
        label_connection.close()

    baseline_test = baseline_test.copy()
    candidate_test = candidate_test.copy()
    baseline_test["variant"] = "baseline"
    candidate_test["variant"] = "rank_target_candidate"
    baseline_test["row_role"] = "test"
    candidate_test["row_role"] = "test"
    unified = pd.concat([baseline_test, candidate_test, *warmup_rows], ignore_index=True)
    unified = unified[["fold_id", "trade_date", "stock_code", "pred_prob", "variant", "row_role"]].sort_values(
        ["fold_id", "variant", "trade_date", "stock_code"], kind="mergesort"
    )
    quality = {
        "rows": int(len(unified)),
        "duplicate_key_groups_by_variant": int(unified.duplicated(["fold_id", "variant", "trade_date", "stock_code"]).sum()),
        "null_pred_prob": int(unified["pred_prob"].isna().sum()),
        "nonfinite_pred_prob": int((~np.isfinite(unified["pred_prob"])).sum()),
        "bj_rows": int(unified["stock_code"].str.endswith(".BJ").sum()),
        "variants": {},
    }
    for variant, frame in unified.groupby("variant", sort=True):
        quality["variants"][variant] = {
            "rows": int(len(frame)),
            "trade_days": int(frame["trade_date"].nunique()),
            "min_trade_date": str(frame["trade_date"].min()),
            "max_trade_date": str(frame["trade_date"].max()),
            "key_count": int(frame[["fold_id", "trade_date", "stock_code"]].drop_duplicates().shape[0]),
        }
    baseline_keys = set(map(tuple, unified.loc[unified["variant"] == "baseline", ["fold_id", "trade_date", "stock_code"]].to_numpy()))
    candidate_keys = set(map(tuple, unified.loc[unified["variant"] == "rank_target_candidate", ["fold_id", "trade_date", "stock_code"]].to_numpy()))
    quality["common_key_count_between_variants"] = int(len(baseline_keys & candidate_keys))
    if any(quality[key] != 0 for key in ("duplicate_key_groups_by_variant", "null_pred_prob", "nonfinite_pred_prob", "bj_rows")):
        raise RuntimeError(f"unified warmup OOF quality failed: {quality}")
    if baseline_keys != candidate_keys:
        raise RuntimeError("baseline/candidate unified OOF key domains differ")

    output_path = output_dir / "baseline_rank_target_warmup_oof.parquet"
    unified.to_parquet(output_path, index=False)
    source_hashes_after = {
        "baseline_oof_sha256": sha256_file(BASELINE_OOF),
        "candidate_oof_sha256": sha256_file(CANDIDATE_OOF),
        "candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST),
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("frozen source OOF asset changed during warmup materialization")
    manifest = {
        "asset_role": "l4_research_strategy_development_warmup_oof_asset",
        "approval_status": "research_only_not_for_l5",
        "frozen_candidate_id": candidate_manifest["candidate_id"],
        "development_closed_after": base.DEVELOPMENT_CLOSED_AFTER,
        "sealed_periods_not_read": ["2025", "2026+"],
        "method": "same frozen baseline and rank-target candidate fold models; embargo-only warmup scoring",
        "warmup_is_not_training": True,
        "warmup_is_not_model_evaluation": True,
        "folds": fold_details,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "quality": quality,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "test_oof_sources_unchanged": True,
        "allow_for_strategy_development_ab_only": True,
        "approved_for_l5": False,
        "allow_next_layer_continue": False,
        "production_unchanged": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(output_dir / "warmup_oof_manifest.json", manifest)
    dump_json(output_dir / "warmup_oof_validation.json", {
        "valid": True,
        "quality": quality,
        "test_oof_sources_unchanged": True,
        "warmup_folds": fold_details,
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    })
    dump_json(output_dir / "hash_inventory.json", {
        "script_sha256": sha256_file(Path(__file__)),
        "unified_warmup_oof_sha256": sha256_file(output_path),
        "warmup_manifest_sha256": sha256_file(output_dir / "warmup_oof_manifest.json"),
        "warmup_validation_sha256": sha256_file(output_dir / "warmup_oof_validation.json"),
        "source_hashes": source_hashes_after,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
