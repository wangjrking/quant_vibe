from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

from ai_module import get_model, incre_fit
from incremental_formal_l4_duckdb_mainline import resolve_savedmodel_feature_aliases
from light_factor_module import build_light_factor_frame, split_light_factor_data
from model_asset_route import resolve_model_feature_duckdb_path, resolve_model_feature_duckdb_table
from model_asset_route import resolve_model_label_duckdb_path, resolve_model_label_duckdb_table
from rolling_train_module import build_rolling_windows


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_latestfold_rerun_compare_20260702"
RUNTIME_DIR = REPORT_DIR / "runtime_selected_features"
SCRIPT_DIR = Path(__file__).resolve().parent


SPECS = [
    {
        "label_key": "1d",
        "label": "executable_1d_open_return",
        "metadata_path": DATA_DIR
        / "reports"
        / "model_agent_1d_fixed4y_latestfold_retrain_v3_20260628"
        / "fixed4y_fs64_k3_t63_m12_lr004_d2_l12"
        / "fold15_17"
        / "models"
        / "model_fold17_metadata.json",
        "test_start": "20260606",
        "test_end": "20260611",
        "train_years": 4,
        "test_months": 3,
        "embargo_days": 1,
        "top_k": 3,
    },
    {
        "label_key": "3d",
        "label": "executable_3d_open_return",
        "metadata_path": DATA_DIR
        / "reports"
        / "model_agent_3d_fixed4y_latestfold_retrain_v1_20260628"
        / "fixed4y_fs64_k3_t126_m8_lr003_d2_l8"
        / "fold15_17"
        / "models"
        / "model_fold17_metadata.json",
        "test_start": "20260606",
        "test_end": "20260609",
        "train_years": 4,
        "test_months": 3,
        "embargo_days": 3,
        "top_k": 3,
    },
    {
        "label_key": "5d",
        "label": "executable_5d_open_return",
        "metadata_path": DATA_DIR
        / "reports"
        / "model_agent_research_5d_fixed4y_fold08_fs120_k10_structure_scan_20260626"
        / "d3_l8_a01_lr004_n5000"
        / "fold08"
        / "models"
        / "model_fold08_metadata.json",
        "test_start": "20260304",
        "test_end": "20260603",
        "train_years": 4,
        "test_months": 3,
        "embargo_days": 11,
        "top_k": 10,
    },
    {
        "label_key": "10d",
        "label": "executable_10d_open_return",
        "metadata_path": DATA_DIR
        / "reports"
        / "model_agent_research_10d_fixed4y_fold08_fs40_structure_scan_20260626"
        / "fs40_d3_l8_lr004_n5000"
        / "fold08"
        / "models"
        / "model_fold08_metadata.json",
        "test_start": "20260304",
        "test_end": "20260603",
        "train_years": 4,
        "test_months": 3,
        "embargo_days": 11,
        "top_k": 10,
    },
]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_status(path: Path, **fields: Any) -> None:
    payload = {"updated_at": now_iso(), **fields}
    write_json(path, payload)


def available_feature_columns() -> tuple[Path, str, set[str]]:
    db_path = resolve_model_feature_duckdb_path(DATA_DIR, require_exists=True)
    table = resolve_model_feature_duckdb_table(DATA_DIR)
    if not table:
        raise RuntimeError("active L3 DuckDB feature table is missing")
    with duckdb.connect(str(db_path), read_only=True) as conn:
        columns = {str(row[0]) for row in conn.execute(f'DESCRIBE "{table}"').fetchall()}
    return db_path, table, columns


def label_input_asset() -> tuple[Path, str]:
    db_path = resolve_model_label_duckdb_path(DATA_DIR, require_exists=True)
    table = resolve_model_label_duckdb_table(DATA_DIR)
    if not table:
        raise RuntimeError("active L3 DuckDB label table is missing")
    return db_path, table


def resolve_runtime_features(requested: list[str], available: set[str]) -> dict[str, Any]:
    resolved = resolve_savedmodel_feature_aliases([str(item) for item in requested], available)
    return {
        "requested_count": len(requested),
        "resolved_count": len(resolved["query_columns"]),
        "resolved_features": resolved["query_columns"],
        "alias_pairs": resolved["alias_pairs"],
        "missing_features": resolved["missing"],
    }


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def load_duckdb_frame(
    *,
    db_path: Path,
    table: str,
    start: str,
    end: str,
    columns: list[str],
) -> pd.DataFrame:
    selected = ", ".join(quote_ident(col) for col in columns)
    sql = (
        f"SELECT {selected} FROM {quote_ident(table)} "
        "WHERE trade_date >= ? AND trade_date <= ? "
        "ORDER BY trade_date, stock_code"
    )
    with duckdb.connect(str(db_path), read_only=True) as conn:
        frame = conn.execute(sql, [start, end]).fetchdf()
    if "trade_date" in frame.columns:
        frame["trade_date"] = frame["trade_date"].astype(str)
    if "stock_code" in frame.columns:
        frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def build_single_window(spec: dict[str, Any]):
    windows = build_rolling_windows(
        data_start="20100104",
        first_test=str(spec["test_start"]),
        final_test=str(spec["test_end"]),
        train_years=int(spec["train_years"]),
        test_months=int(spec["test_months"]),
        step_months=3,
        embargo_days=int(spec["embargo_days"]),
        train_mode="fixed",
    )
    if len(windows) != 1:
        raise RuntimeError(f"expected exactly one rolling window for {spec['label_key']}, got {len(windows)}")
    return windows[0]


def env_from_metadata(metadata: dict[str, Any], *, top_k: int) -> dict[str, str]:
    xgb_params = metadata.get("xgb_params", {})
    sample_weight = metadata.get("sample_weight_config") or {}
    validation = metadata.get("validation_config") or {}
    eval_metric = str(xgb_params.get("eval_metric") or "")
    metric_mode = "top_return_loss" if "top_return_loss" in eval_metric else "custom_mae"
    env = {
        "XGB_REG_EVAL_METRIC": metric_mode,
        "XGB_TOP_RETURN_EVAL_K": str(top_k),
        "XGB_DEVICE": str(xgb_params.get("device") or "cpu"),
        "XGB_N_JOBS": str(int(xgb_params.get("n_jobs") or 8)),
        "XGB_N_ESTIMATORS": str(int(xgb_params.get("n_estimators") or 6000)),
        "XGB_LEARNING_RATE": str(float(xgb_params.get("learning_rate") or 0.003)),
        "XGB_MAX_DEPTH": str(int(xgb_params.get("max_depth") or 3)),
        "XGB_REG_LAMBDA": str(float(xgb_params.get("reg_lambda") or 1.0)),
        "XGB_REG_ALPHA": str(float(xgb_params.get("reg_alpha") or 0.0)),
        "XGB_SUBSAMPLE": str(float(xgb_params.get("subsample") or 1.0)),
        "XGB_COLSAMPLE_BYTREE": str(float(xgb_params.get("colsample_bytree") or 1.0)),
    }
    if xgb_params.get("early_stopping_rounds") is not None:
        env["XGB_EARLY_STOPPING_ROUNDS"] = str(int(xgb_params["early_stopping_rounds"]))
    if sample_weight.get("mode"):
        env["XGB_SAMPLE_WEIGHT_MODE"] = str(sample_weight["mode"])
        env["XGB_SAMPLE_WEIGHT_TOP_PCT"] = str(float(sample_weight["top_pct"]))
        env["XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER"] = str(float(sample_weight["top_multiplier"]))
    if validation.get("mode"):
        env["XGB_VALIDATION_MODE"] = str(validation["mode"])
        env["XGB_VALIDATION_TAIL_DAYS"] = str(int(validation["tail_days"]))
    return env


def apply_fit_env(metadata: dict[str, Any], *, top_k: int, model_path: Path, metadata_path: Path) -> dict[str, str | None]:
    env = env_from_metadata(metadata, top_k=top_k)
    keys = list(env.keys()) + ["XGB_MODEL_SAVE_PATH", "XGB_MODEL_METADATA_PATH"]
    previous = {key: os.environ.get(key) for key in keys}
    for key, value in env.items():
        os.environ[key] = value
    os.environ["XGB_MODEL_SAVE_PATH"] = str(model_path)
    os.environ["XGB_MODEL_METADATA_PATH"] = str(metadata_path)
    return previous


def restore_fit_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def evaluate_predictions(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if "pred_prob" not in frame.columns or label not in frame.columns:
        return {"row_count": int(len(frame)), "trade_days": int(frame["trade_date"].nunique())}
    work = frame.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work["pred_prob"] = pd.to_numeric(work["pred_prob"], errors="coerce")
    work[label] = pd.to_numeric(work[label], errors="coerce")
    work = work.dropna(subset=["pred_prob", label])
    if work.empty:
        return {"row_count": 0, "trade_days": 0}

    daily_rows: list[dict[str, float | str]] = []
    for trade_date, group in work.groupby("trade_date", sort=True):
        scored = group.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        y = scored[label]
        metrics = {
            "trade_date": str(trade_date),
            "rank_ic": float(y.corr(scored["pred_prob"], method="spearman")) if len(scored) > 1 else np.nan,
            "top1": float(y.head(1).mean()),
            "top3": float(y.head(min(3, len(scored))).mean()),
            "top5": float(y.head(min(5, len(scored))).mean()),
            "top10": float(y.head(min(10, len(scored))).mean()),
            "top20": float(y.head(min(20, len(scored))).mean()),
        }
        daily_rows.append(metrics)
    daily = pd.DataFrame(daily_rows)
    return {
        "row_count": int(len(work)),
        "trade_days": int(work["trade_date"].nunique()),
        "date_min": str(work["trade_date"].min()),
        "date_max": str(work["trade_date"].max()),
        "rank_ic": float(daily["rank_ic"].mean()),
        "top1": float(daily["top1"].mean()),
        "top3": float(daily["top3"].mean()),
        "top5": float(daily["top5"].mean()),
        "top10": float(daily["top10"].mean()),
        "top20": float(daily["top20"].mean()),
    }


def resolve_saved_model_path(metadata: dict[str, Any]) -> Path:
    model_path_value = metadata.get("model_path")
    if not model_path_value:
        raise RuntimeError("saved-model metadata missing model_path")
    model_ref = Path(str(model_path_value))
    candidates: list[Path] = []
    if model_ref.is_absolute():
        candidates.append(model_ref)
    else:
        metadata_path_value = metadata.get("metadata_path")
        if metadata_path_value:
            candidates.append((Path(str(metadata_path_value)).resolve().parent / model_ref).resolve())
        candidates.append((SCRIPT_DIR / model_ref).resolve())
        candidates.append((ROOT / model_ref).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "saved-model file not found; tried: " + " | ".join(str(item) for item in candidates)
    )


def predict_with_saved_model(test_x: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    model_ref = resolve_saved_model_path(metadata)
    booster = xgb.Booster()
    booster.load_model(str(model_ref))
    matrix = test_x.apply(pd.to_numeric, errors="coerce").astype("float32")
    return np.asarray(booster.inplace_predict(matrix.to_numpy(copy=False)), dtype="float64")


def compare_predictions(baseline: pd.DataFrame, rerun: pd.DataFrame, label: str) -> dict[str, Any]:
    left = baseline.copy()
    right = rerun.copy()
    left["trade_date"] = left["trade_date"].astype(str)
    right["trade_date"] = right["trade_date"].astype(str)
    key_cols = ["trade_date", "stock_code"]
    merged = left[key_cols + ["pred_prob"]].merge(
        right[key_cols + ["pred_prob"]],
        on=key_cols,
        how="outer",
        suffixes=("_baseline", "_rerun"),
        indicator=True,
    )
    both = merged[merged["_merge"] == "both"].copy()
    baseline_only = int((merged["_merge"] == "left_only").sum())
    rerun_only = int((merged["_merge"] == "right_only").sum())
    diff_summary: dict[str, Any] = {
        "baseline_rows": int(len(left)),
        "rerun_rows": int(len(right)),
        "common_rows": int(len(both)),
        "baseline_only_rows": baseline_only,
        "rerun_only_rows": rerun_only,
        "baseline_eval": evaluate_predictions(left, label),
        "rerun_eval": evaluate_predictions(right, label),
    }
    if both.empty:
        return diff_summary
    both["abs_diff"] = (both["pred_prob_baseline"] - both["pred_prob_rerun"]).abs()
    diff_summary.update(
        {
            "mean_abs_pred_diff": float(both["abs_diff"].mean()),
            "max_abs_pred_diff": float(both["abs_diff"].max()),
            "exact_equal_rows": int((both["abs_diff"] <= 1e-12).sum()),
            "pred_corr": float(both["pred_prob_baseline"].corr(both["pred_prob_rerun"])),
        }
    )
    return diff_summary


def prepare_direct_requested_frames(
    *,
    window,
    label: str,
    requested_features: list[str],
    feature_db_path: Path,
    feature_table: str,
    label_db_path: Path,
    label_table: str,
    available_feature_columns: set[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    resolved = resolve_runtime_features(requested_features, available_feature_columns)
    if resolved["missing_features"]:
        raise RuntimeError(
            f"{label} unresolved direct features: {resolved['missing_features'][:20]}"
        )
    query_columns = list(
        dict.fromkeys(
            ["trade_date", "stock_code", "name", "st_type", "limit_times"]
            + resolved["resolved_features"]
        )
    )
    query_columns = [col for col in query_columns if col in available_feature_columns]
    feature_frame = load_duckdb_frame(
        db_path=feature_db_path,
        table=feature_table,
        start=window.train_start,
        end=window.test_end,
        columns=query_columns,
    )
    for requested_name, source_name in resolved["alias_pairs"]:
        feature_frame[requested_name] = feature_frame[source_name]
    label_frame = load_duckdb_frame(
        db_path=label_db_path,
        table=label_table,
        start=window.train_start,
        end=window.test_end,
        columns=["trade_date", "stock_code", label],
    )
    frame = feature_frame.merge(label_frame, on=["trade_date", "stock_code"], how="inner")
    if "name" in frame.columns:
        frame = frame[frame["name"].notna()]
        frame = frame[~frame["name"].astype(str).str.contains("ST", na=False)]
    if "st_type" in frame.columns:
        frame = frame[frame["st_type"].fillna("").astype(str) != "ST"]
    if "limit_times" in frame.columns:
        frame = frame[frame["limit_times"].isna()]

    train_data = frame[(frame["trade_date"] < window.test_start) & frame[label].notna()].copy()
    test_data = frame[frame["trade_date"] >= window.test_start].copy()
    train_factor_data = train_data[requested_features + [label]].apply(pd.to_numeric, errors="coerce")
    test_factor_data = test_data[requested_features + [label]].apply(pd.to_numeric, errors="coerce")
    train_x = train_factor_data.drop(columns=[label])
    train_y = train_factor_data[label]
    test_x = test_factor_data.drop(columns=[label])
    test_y = test_factor_data[label]
    train_index = pd.MultiIndex.from_frame(train_data[["stock_code", "trade_date"]])
    test_index = pd.MultiIndex.from_frame(test_data[["stock_code", "trade_date"]])
    train_x.index = train_index
    train_y.index = train_index
    test_x.index = test_index
    test_y.index = test_index
    return train_x, train_y, test_x, test_y, train_data, test_data, resolved


def prepare_derived_light_frames(
    *,
    window,
    label: str,
    requested_features: list[str],
    feature_db_path: Path,
    feature_table: str,
    label_db_path: Path,
    label_table: str,
    available_feature_columns: set[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_columns = [
        "trade_date",
        "stock_code",
        "name",
        "st_type",
        "limit_times",
        "industry",
        "industry_encode",
        "stock_encode",
        "amount",
        "vol",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "circ_mv",
        "total_mv",
        "pb",
        "pe",
        "ps",
        "dv_ttm",
        "index_2000_close",
        "index_2000_low",
        "index_2000_high",
        "index_2000_open",
        "index_2000_amount",
        "index_2000_vol",
        "pct_chg",
        "atr_qfq",
        "open_qfq",
        "high_qfq",
        "low_qfq",
        "close_qfq",
        "pre_close_qfq",
    ]
    extra_direct = [col for col in requested_features if col in available_feature_columns]
    query_columns = [col for col in dict.fromkeys(base_columns + extra_direct) if col in available_feature_columns]
    raw_frame = load_duckdb_frame(
        db_path=feature_db_path,
        table=feature_table,
        start=window.train_start,
        end=window.test_end,
        columns=query_columns,
    )
    rename_map = {
        "open_qfq": "open",
        "high_qfq": "high",
        "low_qfq": "low",
        "close_qfq": "close",
        "pre_close_qfq": "pre_close",
    }
    usable_rename = {src: dst for src, dst in rename_map.items() if src in raw_frame.columns}
    raw_frame = raw_frame.rename(columns=usable_rename)
    factor_data = build_light_factor_frame(raw_frame, requested_features, label)
    label_frame = load_duckdb_frame(
        db_path=label_db_path,
        table=label_table,
        start=window.train_start,
        end=window.test_end,
        columns=["trade_date", "stock_code", label],
    )
    factor_data = factor_data.drop(columns=[label], errors="ignore").merge(
        label_frame,
        on=["trade_date", "stock_code"],
        how="left",
    )
    if "name" not in factor_data.columns or factor_data["name"].isna().all():
        factor_data["name"] = factor_data["stock_code"]
    if "st_type" not in factor_data.columns:
        factor_data["st_type"] = ""
    if "limit_times" not in factor_data.columns:
        factor_data["limit_times"] = np.nan
    train_x, train_y, test_x, test_y, train_data, test_data = split_light_factor_data(
        factor_data,
        requested_features,
        label,
        window.train_start,
        window.test_start,
    )
    return train_x, train_y, test_x, test_y, train_data, test_data, {
        "requested_count": len(requested_features),
        "resolved_count": len(requested_features),
        "resolved_features": requested_features,
        "alias_pairs": [
            ["close", "close_qfq"],
            ["high", "high_qfq"],
            ["low", "low_qfq"],
            ["open", "open_qfq"],
            ["pre_close", "pre_close_qfq"],
        ],
        "missing_features": [],
    }


def run_label(spec: dict[str, Any], feature_columns_available: set[str]) -> dict[str, Any]:
    label_key = str(spec["label_key"])
    label = str(spec["label"])
    metadata = read_json(Path(spec["metadata_path"]))
    metadata["metadata_path"] = str(spec["metadata_path"])
    label_dir = REPORT_DIR / label_key
    label_dir.mkdir(parents=True, exist_ok=True)
    status_path = label_dir / "run_status.json"
    write_status(status_path, label_key=label_key, label=label, stage="started")
    feature_db_path, feature_table, _ = available_feature_columns()
    label_db_path, label_table = label_input_asset()
    window = build_single_window(spec)
    runtime_features_path = RUNTIME_DIR / f"{label_key}_features.json"

    requested_features = [str(item) for item in metadata["feature_columns"]]
    try:
        if label_key in {"1d", "3d"}:
            train_x, train_y, test_x, test_y, train_data, test_data, resolved_features = prepare_derived_light_frames(
                window=window,
                label=label,
                requested_features=requested_features,
                feature_db_path=feature_db_path,
                feature_table=feature_table,
                label_db_path=label_db_path,
                label_table=label_table,
                available_feature_columns=feature_columns_available,
            )
        else:
            train_x, train_y, test_x, test_y, train_data, test_data, resolved_features = prepare_direct_requested_frames(
                window=window,
                label=label,
                requested_features=requested_features,
                feature_db_path=feature_db_path,
                feature_table=feature_table,
                label_db_path=label_db_path,
                label_table=label_table,
                available_feature_columns=feature_columns_available,
            )
        write_json(runtime_features_path, {"label": label, "features": list(train_x.columns)})
        write_status(
            status_path,
            label_key=label_key,
            label=label,
            stage="prepared",
            train_rows=int(len(train_x)),
            test_rows=int(len(test_x)),
        )

        model_dir = label_dir / "models"
        pred_path = label_dir / "fold_predictions" / "fold01.parquet"
        model_path = model_dir / "model_fold01.json"
        model_metadata_path = model_dir / "model_fold01_metadata.json"
        previous_env = apply_fit_env(metadata, top_k=int(spec["top_k"]), model_path=model_path, metadata_path=model_metadata_path)
        try:
            write_status(status_path, label_key=label_key, label=label, stage="fit_started")
            model = get_model("reg")
            test_y_out, pred = incre_fit(model, train_x, train_y, test_x, test_y, str(label_dir), save_shap=False)
            write_status(status_path, label_key=label_key, label=label, stage="fit_completed")
        finally:
            restore_fit_env(previous_env)

        rerun = test_data.copy()
        rerun["pred_prob"] = np.asarray(pred, dtype="float64")
        rerun[label] = np.asarray(test_y_out, dtype="float64")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        rerun.to_parquet(pred_path, index=False)

        write_status(status_path, label_key=label_key, label=label, stage="baseline_started")
        baseline = test_data.copy()
        baseline["pred_prob"] = predict_with_saved_model(test_x, metadata)
        baseline[label] = np.asarray(test_y, dtype="float64")
        compare = compare_predictions(baseline, rerun, label)
        payload = {
            "label_key": label_key,
            "label": label,
            "generated_at": now_iso(),
            "metadata_path": str(spec["metadata_path"]),
            "baseline_model_path": str(resolve_saved_model_path(metadata)),
            "rerun_prediction_path": str(pred_path),
            "runtime_selected_features_path": str(runtime_features_path),
            "resolved_feature_summary": {
                "requested_count": resolved_features["requested_count"],
                "resolved_count": resolved_features["resolved_count"],
                "alias_count": len(resolved_features["alias_pairs"]),
                "alias_pairs": resolved_features["alias_pairs"],
            },
            "window": {
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
            },
            "fit_env": env_from_metadata(metadata, top_k=int(spec["top_k"])),
            "train_rows": int(len(train_x)),
            "test_rows": int(len(test_x)),
            "model_output_path": str(model_path),
            "model_metadata_output_path": str(model_metadata_path),
            "compare": compare,
        }
        write_json(label_dir / "compare_summary.json", payload)
        write_status(status_path, label_key=label_key, label=label, stage="completed")
        return payload
    except Exception as exc:
        write_status(
            status_path,
            label_key=label_key,
            label=label,
            stage="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )
        raise


def write_markdown(report_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# 四个标签 latest fold 重训对比",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 当前结论",
        "",
    ]
    for item in payload["labels"]:
        cmp = item["compare"]
        rerun_eval = cmp["rerun_eval"]
        baseline_eval = cmp["baseline_eval"]
        lines.extend(
            [
                f"### {item['label_key']}",
                "",
                f"- 交集行数：`{cmp['common_rows']}`；baseline 独有：`{cmp['baseline_only_rows']}`；重训独有：`{cmp['rerun_only_rows']}`",
                f"- 预测均值绝对差：`{cmp.get('mean_abs_pred_diff', float('nan')):.10f}`；最大绝对差：`{cmp.get('max_abs_pred_diff', float('nan')):.10f}`",
                f"- baseline Full RankIC / Top5：`{baseline_eval.get('rank_ic', float('nan')):.6f}` / `{baseline_eval.get('top5', float('nan')):.6f}`",
                f"- 重训后 Full RankIC / Top5：`{rerun_eval.get('rank_ic', float('nan')):.6f}` / `{rerun_eval.get('top5', float('nan')):.6f}`",
                f"- qfq 显式桥接数：`{item['resolved_feature_summary']['alias_count']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 边界",
            "",
            "- 本次只做 research 重训与预测对比。",
            "- 未发布 production manifest。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重训 latest fold 并对比新旧预测差异")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="仅运行指定标签，例如 --labels 1d 3d",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    feature_db_path, feature_table, feature_columns = available_feature_columns()
    selected = set(args.labels or [str(spec["label_key"]) for spec in SPECS])
    results: list[dict[str, Any]] = []
    for spec in SPECS:
        if str(spec["label_key"]) not in selected:
            continue
        results.append(run_label(spec, feature_columns))

    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_latest_fold_retrain_compare_20260702",
        "input_feature_asset": {
            "db_path": str(feature_db_path),
            "table": feature_table,
        },
        "labels": results,
        "boundaries": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    write_json(REPORT_DIR / "latest_fold_retrain_compare_summary.json", payload)
    write_markdown(REPORT_DIR / "latest_fold_retrain_compare_report.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
