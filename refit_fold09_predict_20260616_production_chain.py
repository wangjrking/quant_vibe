from __future__ import annotations

import argparse
import concurrent.futures
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import xgboost as xgb

from model_asset_route import (
    MODEL_PREDICTION_MODE_INDEPENDENT,
    MODEL_PREDICTION_MODE_LEGACY,
    require_legacy_model_asset_chain_opt_in,
    resolve_legacy_prediction_db_path,
    resolve_model_prediction_db_path,
    resolve_prediction_run_dir,
    use_legacy_prediction_db,
    write_prediction_manifest,
)


LABEL_CONFIGS = {
    "executable_1d_open_return": {
        "dirname": "xgb_reg1d_d3_l3_fs120_exclude_source_limited",
        "train_end": "20260601",
    },
    "executable_3d_open_return": {
        "dirname": "xgb_reg3d_d3_l3_fs120_exclude_source_limited",
        "train_end": "20260530",
    },
    "executable_5d_open_return": {
        "dirname": "xgb_reg5d_d3_l3_fs120_exclude_source_limited",
        "train_end": "20260528",
    },
    "executable_10d_open_return": {
        "dirname": "xgb_reg10d_d3_l3_fs120_exclude_source_limited",
        "train_end": "20260524",
    },
}


def read_dataset(path: Path, columns: list[str], start: str, end: str) -> pd.DataFrame:
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(
        columns=columns,
        filter=(ds.field("trade_date") >= start) & (ds.field("trade_date") <= end),
    )
    frame = table.to_pandas()
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def selected_features(base_report_dir: Path, dirname: str, label: str) -> list[str]:
    path = (
        base_report_dir
        / dirname
        / "feature_scores"
        / dirname
        / f"selected_features_{label}_rolling_fold1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(feature) for feature in payload.get("features", payload)]


def xgb_params(param_registry_dir: Path, label: str) -> dict:
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


def fit_predict_one(job: dict) -> dict:
    started = time.time()
    label = job["label"]
    config = LABEL_CONFIGS[label]
    data_dir = Path(job["data_dir"])
    base_report_dir = Path(job["base_report_dir"])
    output_dir = Path(job["output_dir"])
    predict_date = job["predict_date"]
    production_dir = data_dir / "production_factor_parts"
    label_dir = data_dir / "prediction_label_parts"
    param_registry_dir = data_dir / "model_param_registry"

    prod_schema = set(ds.dataset(str(production_dir), format="parquet").schema.names)
    label_schema = set(ds.dataset(str(label_dir), format="parquet").schema.names)
    if label not in label_schema:
        raise RuntimeError(f"label not found in prediction_label_parts: {label}")

    requested_features = selected_features(base_report_dir, config["dirname"], label)
    features = [feature for feature in requested_features if feature in prod_schema]
    dropped_features = [feature for feature in requested_features if feature not in prod_schema]
    if not features:
        raise RuntimeError(f"no selected features are available in production_factor_parts for {label}")

    production_columns = ["stock_code", "trade_date", *features]
    train_features = read_dataset(production_dir, production_columns, "20100101", config["train_end"])
    train_labels = read_dataset(label_dir, ["stock_code", "trade_date", label], "20100101", config["train_end"])
    train = train_features.merge(train_labels, on=["stock_code", "trade_date"], how="inner", validate="one_to_one")
    train = train.dropna(subset=[label])
    predict = read_dataset(production_dir, production_columns, predict_date, predict_date)
    if train.empty or predict.empty:
        raise RuntimeError(f"{label} empty train or predict frame: train={len(train)} predict={len(predict)}")

    train_x = train[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    train_y = pd.to_numeric(train[label], errors="coerce").astype("float32")
    predict_x = predict[features].apply(pd.to_numeric, errors="coerce").astype("float32")

    model = xgb.XGBRegressor(**xgb_params(param_registry_dir, label))
    model.fit(train_x, train_y, verbose=100)
    pred = model.predict(predict_x)

    label_output_dir = output_dir / config["dirname"]
    model_dir = label_output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_fold09_production_chain_{predict_date}.json"
    metadata_path = model_dir / f"model_fold09_production_chain_{predict_date}_metadata.json"
    prediction_path = label_output_dir / f"prediction_{label}_{predict_date}.parquet"
    label_output_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_path))
    metadata = {
        "label": label,
        "model_path": str(model_path),
        "feature_source": str(production_dir),
        "label_source": str(label_dir),
        "train_start": "20100101",
        "train_end": config["train_end"],
        "predict_date": predict_date,
        "requested_fold1_feature_count": len(requested_features),
        "used_feature_count": len(features),
        "dropped_features_not_in_production_chain": dropped_features,
        "features": features,
        "xgb_params": model.get_params(),
        "train_rows": int(len(train)),
        "predict_rows": int(len(predict)),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    out = predict[["stock_code", "trade_date"]].copy()
    out["label"] = label
    out["model_fold"] = 9
    out["pred_prob"] = pred.astype("float64")
    out["model_path"] = str(model_path)
    out["feature_source"] = str(production_dir)
    out.to_parquet(prediction_path, index=False)

    return {
        "label": label,
        "status": "ok",
        "train_rows": int(len(train)),
        "predict_rows": int(len(predict)),
        "used_feature_count": len(features),
        "dropped_feature_count": len(dropped_features),
        "prediction_path": str(prediction_path),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "elapsed_sec": round(time.time() - started, 2),
    }


def write_combined(
    output_dir: Path,
    table_name: str,
    data_dir: Path,
    results: list[dict],
    *,
    prediction_output_mode: str | None = None,
) -> dict:
    frames = [pd.read_parquet(result["prediction_path"]) for result in results]
    long_frame = pd.concat(frames, ignore_index=True)
    long_path = output_dir / f"{table_name}.parquet"
    long_frame.to_parquet(long_path, index=False)

    base = long_frame[["stock_code", "trade_date"]].drop_duplicates().copy()
    for result in results:
        label = result["label"]
        pred_col = "pred_" + label.replace("executable_", "").replace("_open_return", "")
        pred = long_frame[long_frame["label"] == label][["stock_code", "trade_date", "pred_prob"]].rename(
            columns={"pred_prob": pred_col}
        )
        base = base.merge(pred, on=["stock_code", "trade_date"], how="left")
    wide_path = output_dir / f"{table_name}_wide.parquet"
    base.to_parquet(wide_path, index=False)

    if use_legacy_prediction_db(prediction_output_mode):
        require_legacy_model_asset_chain_opt_in(reason="legacy odb production-chain combined prediction output")
        db_path = resolve_legacy_prediction_db_path(data_dir)
        prediction_mode = MODEL_PREDICTION_MODE_LEGACY
    else:
        db_path = resolve_model_prediction_db_path(data_dir, create_parent=True)
        prediction_mode = MODEL_PREDICTION_MODE_INDEPENDENT
    conn = sqlite3.connect(db_path, timeout=300)
    try:
        conn.execute("PRAGMA busy_timeout=300000")
        long_frame.to_sql(table_name, conn, if_exists="replace", index=False)
        base.to_sql(f"{table_name}_wide", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()

    run_dir = resolve_prediction_run_dir(data_dir, label="production_chain", output_table=table_name, create=True)
    write_prediction_manifest(
        run_dir / "prediction_manifest.json",
        {
            "label_scope": "production_chain_multi_label",
            "prediction_mode": prediction_mode,
            "prediction_db_path": str(db_path),
            "prediction_table_long": table_name,
            "prediction_table_wide": f"{table_name}_wide",
            "long_rows": int(len(long_frame)),
            "wide_rows": int(len(base)),
            "source_output_dir": str(output_dir),
        },
    )

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
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--labels", default=",".join(LABEL_CONFIGS.keys()))
    parser.add_argument(
        "--prediction-output-mode",
        default=None,
        choices=[MODEL_PREDICTION_MODE_INDEPENDENT, MODEL_PREDICTION_MODE_LEGACY],
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    base_report_dir = data_dir / "reports" / "model_agent_standard_factors_20260616"
    output_dir = base_report_dir / f"fold09_production_chain_predict_{args.predict_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    unknown = [label for label in requested_labels if label not in LABEL_CONFIGS]
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")

    common = {
        "data_dir": str(data_dir),
        "base_report_dir": str(base_report_dir),
        "output_dir": str(output_dir),
        "predict_date": args.predict_date,
    }
    jobs = [dict(common, label=label) for label in requested_labels]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(fit_predict_one, job): job["label"] for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            label = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"label": label, "status": "failed", "error": repr(exc)}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)

    payload = {"predict_date": args.predict_date, "results": results}
    if all(result.get("status") == "ok" for result in results):
        table_name = f"stock_predict_data_model_agent_production_chain_{args.predict_date}"
        payload["combined"] = write_combined(
            output_dir,
            table_name,
            data_dir,
            sorted(results, key=lambda x: x["label"]),
            prediction_output_mode=args.prediction_output_mode,
        )
        payload["status"] = "ok"
    else:
        payload["status"] = "failed"
    status_path = output_dir / "run_status.json"
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status_path": str(status_path), **payload}, ensure_ascii=False), flush=True)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
