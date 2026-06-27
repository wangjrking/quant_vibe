from __future__ import annotations

import argparse
import concurrent.futures
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xgboost as xgb


BASE_OUTPUT_COLUMNS = [
    "trade_date",
    "stock_code",
    "name",
    "std_his_high",
    "10d_yield_rate",
    "2d_yield_rate",
    "st_type",
    "post_high",
    "post_close",
    "post2_close",
    "post2_high",
    "open3_yield_rate",
    "open2_yield_rate",
    "limit_times",
    "close",
    "pre_close",
    "post_open",
    "post2_open",
    "post3_open",
    "post4_open",
    "post5_open",
    "post6_open",
    "post12_open",
    "industry",
    "industry_encode",
    "atr_qfq",
    "close_rate",
    "amount",
    "vol",
    "turnover_rate",
    "turnover_rate_f",
    "circ_mv",
    "total_mv",
    "volume_ratio",
]


LABEL_CONFIGS = {
    "executable_1d_open_return": {
        "dirname": "xgb_reg1d_d3_l3_fs120_exclude_source_limited",
        "train_end": "20260601",
        "required_label_cols": ["post_open", "post2_open"],
    },
    "executable_3d_open_return": {
        "dirname": "xgb_reg3d_d3_l3_fs120_exclude_source_limited",
        "train_end": "20260530",
        "required_label_cols": ["post_open", "post4_open"],
    },
    "executable_5d_open_return": {
        "dirname": "xgb_reg5d_d3_l3_fs120_exclude_source_limited",
        "train_end": "20260528",
        "required_label_cols": ["post_open", "post6_open"],
    },
    "executable_10d_open_return": {
        "dirname": "xgb_reg10d_d3_l3_fs120_exclude_source_limited",
        "train_end": "20260524",
        "required_label_cols": ["post_open", "post12_open"],
    },
}


def _read_dataset(path: Path, columns: list[str], start: str, end: str) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=["stock_code", "trade_date"])
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(
        columns=columns,
        filter=(ds.field("trade_date") >= start) & (ds.field("trade_date") <= end),
    )
    frame = table.to_pandas()
    if "trade_date" in frame.columns:
        frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def _selected_features(base_report_dir: Path, dirname: str, label: str) -> list[str]:
    path = (
        base_report_dir
        / dirname
        / "feature_scores"
        / dirname
        / f"selected_features_{label}_rolling_fold1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(feature) for feature in payload.get("features", payload)]


def _label_values(frame: pd.DataFrame, label: str) -> pd.Series:
    buy = pd.to_numeric(frame["post_open"], errors="coerce")
    if label == "executable_1d_open_return":
        sell = pd.to_numeric(frame["post2_open"], errors="coerce")
    elif label == "executable_3d_open_return":
        sell = pd.to_numeric(frame["post4_open"], errors="coerce")
    elif label == "executable_5d_open_return":
        sell = pd.to_numeric(frame["post6_open"], errors="coerce")
    elif label == "executable_10d_open_return":
        sell = pd.to_numeric(frame["post12_open"], errors="coerce")
    else:
        raise ValueError(f"unsupported label: {label}")
    entry_cash = buy * (1.0 + 0.0003 + 0.001)
    exit_cash = sell * (1.0 - 0.0003 - 0.0005 - 0.001)
    return exit_cash / entry_cash - 1.0


def _xgb_params(param_registry_dir: Path, label: str) -> dict:
    payload = json.loads((param_registry_dir / f"{label}.json").read_text(encoding="utf-8"))
    params = payload["current_candidate"]["xgb_params"]
    return {
        "objective": params.get("objective", "reg:squarederror"),
        "n_estimators": int(params.get("n_estimators", 3000)),
        "learning_rate": float(params.get("learning_rate", 0.005)),
        "max_depth": int(params.get("max_depth", 3)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "reg_alpha": float(params.get("reg_alpha", 0)),
        "reg_lambda": float(params.get("reg_lambda", 3)),
        "random_state": int(params.get("random_state", 42)),
        "device": params.get("device", "cuda"),
        "n_jobs": int(params.get("n_jobs", 2)),
    }


def _fit_predict_one(args: dict) -> dict:
    started = time.time()
    label = args["label"]
    config = LABEL_CONFIGS[label]
    base_report_dir = Path(args["base_report_dir"])
    clean_dir = Path(args["clean_dir"])
    raw_dir = Path(args["raw_dir"])
    output_dir = Path(args["output_dir"])
    param_registry_dir = Path(args["param_registry_dir"])
    predict_date = args["predict_date"]

    dirname = config["dirname"]
    train_end = config["train_end"]
    features = _selected_features(base_report_dir, dirname, label)

    clean_schema = set(ds.dataset(str(clean_dir), format="parquet").schema.names)
    raw_schema = set(ds.dataset(str(raw_dir), format="parquet").schema.names)
    required = set(["stock_code", "trade_date", *features, *config["required_label_cols"]])
    optional = set(BASE_OUTPUT_COLUMNS) - required
    needed = required | optional
    missing = sorted(column for column in required if column not in clean_schema and column not in raw_schema)
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")

    clean_columns = sorted(column for column in needed if column in clean_schema)
    raw_columns = sorted(column for column in needed if column not in clean_schema and column in raw_schema)
    clean = _read_dataset(clean_dir, clean_columns, "20100101", predict_date)
    raw = _read_dataset(raw_dir, ["stock_code", "trade_date", *raw_columns], "20100101", predict_date)
    if raw_columns:
        frame = clean.merge(raw, on=["stock_code", "trade_date"], how="left", validate="one_to_one")
    else:
        frame = clean

    frame[label] = _label_values(frame, label)
    train = frame[(frame["trade_date"] >= "20100101") & (frame["trade_date"] <= train_end)].copy()
    predict = frame[frame["trade_date"] == predict_date].copy()
    train = train.dropna(subset=[label])
    if train.empty or predict.empty:
        raise RuntimeError(f"{label} empty train or predict frame: train={len(train)} predict={len(predict)}")

    for feature in features:
        train[feature] = pd.to_numeric(train[feature], errors="coerce")
        predict[feature] = pd.to_numeric(predict[feature], errors="coerce")

    train_x = train[features].astype("float32")
    train_y = pd.to_numeric(train[label], errors="coerce").astype("float32")
    predict_x = predict[features].astype("float32")

    model = xgb.XGBRegressor(**_xgb_params(param_registry_dir, label))
    model.fit(train_x, train_y, verbose=100)
    pred = model.predict(predict_x)

    output_label_dir = output_dir / dirname
    model_dir = output_label_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_fold09_refit_{predict_date}.json"
    metadata_path = model_dir / f"model_fold09_refit_{predict_date}_metadata.json"
    prediction_path = output_label_dir / f"prediction_{label}_{predict_date}.parquet"
    output_label_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_path))
    metadata = {
        "label": label,
        "model_path": str(model_path),
        "source_factor_dir": str(clean_dir),
        "raw_auxiliary_dir": str(raw_dir),
        "train_start": "20100101",
        "train_end": train_end,
        "predict_date": predict_date,
        "feature_source": "fold1 selected features from model_agent_standard_factors_20260616",
        "feature_count": len(features),
        "features": features,
        "xgb_params": model.get_params(),
        "train_rows": int(len(train)),
        "predict_rows": int(len(predict)),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    out = predict.copy()
    out["pred_prob"] = pred.astype("float64")
    out["label"] = label
    out["model_fold"] = 9
    out["model_path"] = str(model_path)
    for column in BASE_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    keep = ["label", "model_fold", *BASE_OUTPUT_COLUMNS, "pred_prob", "model_path"]
    out[keep].to_parquet(prediction_path, index=False)

    return {
        "label": label,
        "status": "ok",
        "train_rows": int(len(train)),
        "predict_rows": int(len(predict)),
        "prediction_path": str(prediction_path),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "elapsed_sec": round(time.time() - started, 2),
    }


def _write_combined(output_dir: Path, table_name: str, data_dir: Path, results: list[dict]) -> dict:
    frames = [pd.read_parquet(result["prediction_path"]) for result in results if result.get("status") == "ok"]
    if not frames:
        raise RuntimeError("no successful prediction frames")
    long_frame = pd.concat(frames, ignore_index=True)
    long_path = output_dir / f"{table_name}.parquet"
    long_frame.to_parquet(long_path, index=False)

    wide_base_cols = [column for column in BASE_OUTPUT_COLUMNS if column in long_frame.columns]
    base = long_frame[long_frame["label"] == results[0]["label"]][wide_base_cols].copy()
    for result in results:
        label = result["label"]
        pred_col = "pred_" + label.replace("executable_", "").replace("_open_return", "")
        pred = long_frame[long_frame["label"] == label][["stock_code", "trade_date", "pred_prob"]].rename(
            columns={"pred_prob": pred_col}
        )
        base = base.merge(pred, on=["stock_code", "trade_date"], how="outer")
    wide_path = output_dir / f"{table_name}_wide.parquet"
    base.to_parquet(wide_path, index=False)

    db_path = data_dir / "odb.db"
    conn = sqlite3.connect(db_path, timeout=300)
    try:
        conn.execute("PRAGMA busy_timeout=300000")
        long_frame.to_sql(table_name, conn, if_exists="replace", index=False)
        base.to_sql(f"{table_name}_wide", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()

    return {
        "long_parquet": str(long_path),
        "wide_parquet": str(wide_path),
        "sqlite_db": str(db_path),
        "sqlite_table_long": table_name,
        "sqlite_table_wide": f"{table_name}_wide",
        "long_rows": int(len(long_frame)),
        "wide_rows": int(len(base)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=r"D:\work\quant\quant_mcp\quant\data_file")
    parser.add_argument("--predict-date", default="20260616")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--labels", default=",".join(LABEL_CONFIGS.keys()))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    base_report_dir = data_dir / "reports" / "model_agent_standard_factors_20260616"
    output_dir = base_report_dir / f"fold09_refit_predict_{args.predict_date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "base_report_dir": str(base_report_dir),
        "clean_dir": str(data_dir / "production_factor_parts_clean_20260616"),
        "raw_dir": str(data_dir / "raw_factor_by_stock_parts"),
        "output_dir": str(output_dir),
        "param_registry_dir": str(data_dir / "model_param_registry"),
        "predict_date": args.predict_date,
    }
    requested_labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    unknown = [label for label in requested_labels if label not in LABEL_CONFIGS]
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    jobs = [dict(common, label=label) for label in requested_labels]
    results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(_fit_predict_one, job): job["label"] for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            label = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"label": label, "status": "failed", "error": repr(exc)}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)

    status_path = output_dir / "run_status.json"
    payload = {"predict_date": args.predict_date, "results": results}
    if all(result.get("status") == "ok" for result in results):
        table_name = f"stock_predict_data_model_agent_fold09_refit_{args.predict_date}"
        all_results = list(results)
        for label, config in LABEL_CONFIGS.items():
            if any(result.get("label") == label for result in all_results):
                continue
            prediction_path = (
                output_dir
                / config["dirname"]
                / f"prediction_{label}_{args.predict_date}.parquet"
            )
            model_path = (
                output_dir
                / config["dirname"]
                / "models"
                / f"model_fold09_refit_{args.predict_date}.json"
            )
            metadata_path = (
                output_dir
                / config["dirname"]
                / "models"
                / f"model_fold09_refit_{args.predict_date}_metadata.json"
            )
            if prediction_path.exists() and model_path.exists():
                all_results.append(
                    {
                        "label": label,
                        "status": "ok",
                        "prediction_path": str(prediction_path),
                        "model_path": str(model_path),
                        "metadata_path": str(metadata_path),
                        "reused_from_previous_run": True,
                    }
                )
        if all(label in {result.get("label") for result in all_results if result.get("status") == "ok"} for label in LABEL_CONFIGS):
            payload["combined"] = _write_combined(
                output_dir,
                table_name,
                data_dir,
                sorted(all_results, key=lambda x: x["label"]),
            )
            payload["status"] = "ok"
        else:
            payload["status"] = "partial_ok"
    else:
        payload["status"] = "failed"
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status_path": str(status_path), **payload}, ensure_ascii=False), flush=True)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
