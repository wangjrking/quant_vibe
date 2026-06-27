from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xgboost as xgb

from model_asset_route import (
    enrich_research_prediction_manifest,
    resolve_model_prediction_db_path,
    resolve_prediction_run_dir,
    write_prediction_manifest,
)


LABEL_MODEL_SPECS = {
    "executable_1d_open_return": {
        "model_dir": "xgb_reg1d_d2_l5_fs80_standardchain_fold1fixed_20260617",
        "model_note": "latest completed standard-chain 1d model; current short-cycle v2 run is still in progress",
    },
    "executable_3d_open_return": {
        "model_dir": "xgb_reg3d_d3_l5_fs160_lr003_n6000_standardchain_fold1fixed_20260617",
        "model_note": "latest completed standard-chain 3d model",
    },
    "executable_5d_open_return": {
        "model_dir": "xgb_reg5d_d3_l3_fs160_lr003_n6000_standardchain_fold1fixed_20260617",
        "model_note": "latest completed standard-chain 5d model",
    },
    "executable_10d_open_return": {
        "model_dir": "xgb_reg10d_d3_l3_fs160_lr003_n6000_standardchain_fold1fixed_20260617",
        "model_note": "latest completed standard-chain 10d model",
    },
}

BASE_COLUMNS = [
    "stock_code",
    "trade_date",
    "name",
    "industry",
    "circ_mv",
    "total_mv",
    "turnover_rate",
    "turnover_rate_f",
]


def _label_suffix(label: str) -> str:
    return label.replace("executable_", "").replace("_open_return", "")


def _metadata_path(report_root: Path, model_dir: str) -> Path:
    return report_root / model_dir / "models" / "model_fold09_metadata.json"


def _model_path(report_root: Path, model_dir: str) -> Path:
    return report_root / model_dir / "models" / "model_fold09.json"


def _read_target_date_frame(factor_dir: Path, predict_date: str, columns: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(str(factor_dir), format="parquet")
    schema_names = set(dataset.schema.names)
    columns = [column for column in columns if column in schema_names]
    table = dataset.to_table(
        columns=columns,
        filter=(ds.field("trade_date") == predict_date),
    )
    frame = table.to_pandas()
    if "trade_date" in frame.columns:
        frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def _load_model_metadata(report_root: Path, label: str) -> dict:
    spec = LABEL_MODEL_SPECS[label]
    meta_path = _metadata_path(report_root, spec["model_dir"])
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["model_note"] = spec["model_note"]
    payload["metadata_path"] = str(meta_path)
    payload["model_path"] = str(_model_path(report_root, spec["model_dir"]))
    payload["model_dir"] = spec["model_dir"]
    return payload


def _predict_one(frame: pd.DataFrame, label: str, metadata: dict) -> pd.DataFrame:
    features = [str(col) for col in metadata["feature_columns"]]
    booster = xgb.Booster()
    booster.load_model(metadata["model_path"])
    x = frame[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    pred = np.asarray(booster.inplace_predict(x.to_numpy(copy=False)), dtype="float64")
    out = frame[[col for col in BASE_COLUMNS if col in frame.columns]].copy()
    suffix = _label_suffix(label)
    out[f"pred_{suffix}"] = pred
    out[f"rank_pct_{suffix}"] = out[f"pred_{suffix}"].rank(method="average", pct=True, ascending=False)
    out[f"model_dir_{suffix}"] = metadata["model_dir"]
    out[f"model_path_{suffix}"] = metadata["model_path"]
    return out


def _merge_label_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0].copy()
    key_cols = [col for col in BASE_COLUMNS if col in merged.columns]
    for frame in frames[1:]:
        add_cols = [col for col in frame.columns if col not in key_cols]
        merged = merged.merge(frame[["stock_code", "trade_date", *add_cols]], on=["stock_code", "trade_date"], how="outer")
    return merged


def _long_from_wide(wide: pd.DataFrame, labels: list[str], metadata_by_label: dict[str, dict]) -> pd.DataFrame:
    long_frames = []
    base_cols = [col for col in BASE_COLUMNS if col in wide.columns]
    for label in labels:
        suffix = _label_suffix(label)
        pred_col = f"pred_{suffix}"
        rank_col = f"rank_pct_{suffix}"
        frame = wide[base_cols + [pred_col, rank_col]].copy()
        frame["label"] = label
        frame["pred_prob"] = frame.pop(pred_col)
        frame["score_rank_pct"] = frame.pop(rank_col)
        frame["model_dir"] = metadata_by_label[label]["model_dir"]
        frame["model_path"] = metadata_by_label[label]["model_path"]
        frame["metadata_path"] = metadata_by_label[label]["metadata_path"]
        frame["model_note"] = metadata_by_label[label]["model_note"]
        long_frames.append(frame)
    return pd.concat(long_frames, ignore_index=True)


def _write_sqlite_tables(db_path: Path, wide_table: str, long_table: str, wide: pd.DataFrame, long: pd.DataFrame) -> None:
    with sqlite3.connect(db_path, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        wide.to_sql(wide_table, conn, if_exists="replace", index=False)
        long.to_sql(long_table, conn, if_exists="replace", index=False)
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a target date with saved standard-chain model files only.")
    parser.add_argument("--data-dir", default=r"D:\work\quant\quant_mcp\quant\data_file")
    parser.add_argument("--predict-date", required=True)
    parser.add_argument("--labels", default=",".join(LABEL_MODEL_SPECS.keys()))
    parser.add_argument("--report-root", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report_root = Path(args.report_root) if args.report_root else data_dir / "reports" / "model_agent_standard_chain_tune_20260617"
    factor_dir = data_dir / "production_factor_parts"
    output_dir = data_dir / "reports" / f"saved_model_predict_{args.predict_date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    unknown = [label for label in labels if label not in LABEL_MODEL_SPECS]
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")

    metadata_by_label = {label: _load_model_metadata(report_root, label) for label in labels}
    all_feature_columns = sorted(
        {
            *BASE_COLUMNS,
            *[
                feature
                for metadata in metadata_by_label.values()
                for feature in metadata["feature_columns"]
            ],
        }
    )
    frame = _read_target_date_frame(factor_dir, args.predict_date, all_feature_columns)
    if frame.empty:
        raise RuntimeError(f"no production_factor_parts rows found for {args.predict_date}")

    label_frames = [_predict_one(frame, label, metadata_by_label[label]) for label in labels]
    wide = _merge_label_frames(label_frames)
    wide = wide.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    long = _long_from_wide(wide, labels, metadata_by_label)
    long = long.sort_values(["label", "trade_date", "stock_code"]).reset_index(drop=True)

    wide_table = f"stock_predict_data_saved_model_scores_{args.predict_date}_wide"
    long_table = f"stock_predict_data_saved_model_scores_{args.predict_date}_long"
    db_path = resolve_model_prediction_db_path(data_dir, create_parent=True)
    _write_sqlite_tables(db_path, wide_table, long_table, wide, long)

    wide_path = output_dir / f"{wide_table}.parquet"
    long_path = output_dir / f"{long_table}.parquet"
    wide.to_parquet(wide_path, index=False)
    long.to_parquet(long_path, index=False)

    manifest_dir = resolve_prediction_run_dir(data_dir, label="saved_model_scores", output_table=wide_table, create=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = enrich_research_prediction_manifest(
        {
            "schema_version": 1,
            "source_type": "sqlite_table",
            "predict_date": args.predict_date,
            "prediction_db_path": str(db_path),
            "db_path": "../MODEL_PREDICTIONS.db",
            "table": wide_table,
            "table_long": long_table,
            "prediction_mode": "independent",
            "row_count": int(len(wide)),
            "long_row_count": int(len(long)),
            "trade_days": int(wide["trade_date"].nunique()),
            "stock_count": int(wide["stock_code"].nunique()),
            "min_trade_date": str(wide["trade_date"].min()),
            "max_trade_date": str(wide["trade_date"].max()),
            "market_db_path": "../../STOCK_DAILY_DATA.db",
            "output_dir": str(output_dir),
            "wide_parquet": str(wide_path),
            "long_parquet": str(long_path),
            "generated_at": generated_at,
            "labels": labels,
            "label_model_specs": {
                label: {
                    "model_dir": metadata["model_dir"],
                    "model_path": metadata["model_path"],
                    "metadata_path": metadata["metadata_path"],
                    "feature_count": len(metadata["feature_columns"]),
                    "model_note": metadata["model_note"],
                }
                for label, metadata in metadata_by_label.items()
            },
            "notes": "Saved-model single-date prediction on the standard production factor chain. No retraining or model refit was performed.",
        }
    )
    manifest_path = write_prediction_manifest(manifest_dir / "prediction_manifest.json", payload)

    status_payload = {
        "status": "ok",
        "predict_date": args.predict_date,
        "prediction_db_path": str(db_path),
        "wide_table": wide_table,
        "long_table": long_table,
        "wide_rows": int(len(wide)),
        "long_rows": int(len(long)),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
    }
    (output_dir / "run_status.json").write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
