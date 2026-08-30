from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

from model_asset_route import (
    enrich_research_prediction_manifest,
    resolve_model_feature_duckdb_path,
    resolve_model_feature_duckdb_table,
    resolve_model_prediction_duckdb_path,
    resolve_prediction_run_dir,
    write_prediction_manifest,
)
from stock_daily_data_route import resolve_stock_daily_duckdb_path


BASE_OUTPUT_COLUMNS = [
    "trade_date",
    "stock_code",
    "pred_prob",
    "10d_yield_rate",
    "close",
    "pre_close",
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


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _read_duckdb_table(db_path: Path, table: str, start: str, end: str) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        existing = {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}
        if table not in existing:
            raise RuntimeError(f"base table not found: {db_path}::{table}")
        query = f"""
            select *
            from {_quote(table)}
            where trade_date >= ? and trade_date <= ?
            order by trade_date, stock_code
        """
        frame = conn.execute(query, [start, end]).fetchdf()
    if frame.empty:
        raise RuntimeError(f"base table has no rows in requested range: {start}-{end}")
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def _date_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame.groupby("trade_date", sort=True).size().items()}


def _dup_count(frame: pd.DataFrame) -> int:
    return int(frame.groupby(["stock_code", "trade_date"], sort=False).size().gt(1).sum())


def _manifest_rel_path(path: Path, manifest_dir: Path) -> str:
    return Path(os.path.relpath(path, start=manifest_dir)).as_posix()


def _read_factor_dates_duckdb(
    db_path: Path,
    table: str,
    dates: list[str],
    columns: list[str],
    *,
    optional_columns: set[str] | None = None,
) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        available_columns = {str(row[0]) for row in conn.execute(f"DESCRIBE {_quote(table)}").fetchall()}
        optional_columns = optional_columns or set()
        read_columns = [column for column in columns if column in available_columns]
        missing = sorted(set(columns) - set(read_columns) - optional_columns)
        if missing:
            raise RuntimeError(f"production_factor_parts DuckDB table is missing required columns: {missing[:20]}")
        selected = ", ".join(_quote(column) for column in read_columns)
        sql = (
            f"SELECT {selected} "
            f"FROM {_quote(table)} "
            "WHERE trade_date IN (SELECT unnest(?)) "
            "ORDER BY trade_date, stock_code"
        )
        frame = conn.execute(sql, [[str(date) for date in dates]]).fetchdf()
    if frame.empty:
        raise RuntimeError(f"no production_factor_parts DuckDB rows found for dates: {dates}")
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def read_registered_factor_dates(
    data_dir: Path,
    dates: list[str],
    columns: list[str],
    *,
    optional_columns: set[str] | None = None,
) -> pd.DataFrame:
    feature_table = resolve_model_feature_duckdb_table(data_dir)
    if not feature_table:
        raise RuntimeError("active L3 DuckDB feature table is required; Parquet fallback is disabled")
    feature_db_path = resolve_model_feature_duckdb_path(data_dir, require_exists=True)
    return _read_factor_dates_duckdb(
        feature_db_path,
        feature_table,
        dates,
        columns,
        optional_columns=optional_columns,
    )


def _load_metadata(model_dir: Path, fold: int) -> dict:
    metadata_path = model_dir / "models" / f"model_fold{fold:02d}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = Path(metadata.get("model_path") or model_dir / "models" / f"model_fold{fold:02d}.json")
    if not model_path.is_absolute():
        model_path = model_dir / "models" / model_path.name
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    metadata["metadata_path"] = str(metadata_path)
    metadata["model_path"] = str(model_path)
    return metadata


def _predict_incremental(frame: pd.DataFrame, *, metadata: dict, label: str, output_columns: list[str]) -> pd.DataFrame:
    features = [str(column) for column in metadata["feature_columns"]]
    booster = xgb.Booster()
    booster.load_model(metadata["model_path"])
    x = frame[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    pred = np.asarray(booster.inplace_predict(x.to_numpy(copy=False)), dtype="float64")
    out = frame[[column for column in output_columns if column in frame.columns and column != "pred_prob"]].copy()
    out["pred_prob"] = pred
    if label in output_columns and label not in out.columns:
        out[label] = np.nan
    for column in output_columns:
        if column not in out.columns:
            out[column] = np.nan
    return out[output_columns]


def _write_duckdb_table(db_path: Path, table: str, frame: pd.DataFrame) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as conn:
        conn.register("_score_df", frame)
        conn.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _score_df')
        conn.unregister("_score_df")


def _manifest_payload(
    *,
    label: str,
    candidate_id: str,
    output_table: str,
    base_table: str,
    frame: pd.DataFrame,
    model_metadata: dict,
    incremental_dates: list[str],
    generated_at: str,
    output_db_path: Path,
    manifest_dir: Path,
) -> dict:
    source_type = "duckdb_table"
    db_path = _manifest_rel_path(output_db_path, manifest_dir)
    market_db_path = _manifest_rel_path(resolve_stock_daily_duckdb_path(), manifest_dir)
    return enrich_research_prediction_manifest(
        {
            "schema_version": 1,
            "asset_role": "l4_research_prediction_asset",
            "model_track": "research_score_asset",
            "approval_status": "research_only_not_for_l5",
            "promotion_requires_user_confirmation": True,
            "source_type": source_type,
            "label": label,
            "candidate_id": candidate_id,
            "db_path": db_path,
            "table": output_table,
            "market_db_path": market_db_path,
            "generated_at": generated_at,
            "row_count": int(len(frame)),
            "trade_days": int(frame["trade_date"].nunique()),
            "stock_count": int(frame["stock_code"].nunique()),
            "min_trade_date": str(frame["trade_date"].min()),
            "max_trade_date": str(frame["trade_date"].max()),
            "duplicate_keys": _dup_count(frame),
            "base_training_eval_table": base_table,
            "incremental_score_dates": incremental_dates,
            "model_path": model_metadata["model_path"],
            "model_metadata_path": model_metadata["metadata_path"],
            "feature_count": int(len(model_metadata["feature_columns"])),
            "notes": (
                "Standard L4 strategy score asset. Historical rows come from the rolling evaluation "
                "prediction table; incremental rows are saved-model inference from production_factor_parts. "
                "No retraining was performed by this score build step."
            ),
        }
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standard-chain strategy score asset from rolling predictions and saved fold model.")
    parser.add_argument("--data-dir", default=r"D:\work\quant\quant_mcp\quant\data_file")
    parser.add_argument("--label", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--model-dir", required=True, help="Directory containing models/model_foldXX.json and metadata")
    parser.add_argument("--base-table", required=True, help="Merged rolling evaluation prediction table")
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--score-start", default="20240604")
    parser.add_argument("--score-end", required=True)
    parser.add_argument("--model-fold", type=int, default=9)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    db_path = resolve_model_prediction_duckdb_path(data_dir, require_exists=False)
    model_dir = Path(args.model_dir)
    model_metadata = _load_metadata(model_dir, args.model_fold)
    base = _read_duckdb_table(db_path, args.base_table, args.score_start, args.score_end)
    base_max = str(base["trade_date"].max())
    output_columns = [column for column in BASE_OUTPUT_COLUMNS if column in base.columns]
    if args.label in base.columns:
        output_columns.append(args.label)
    feature_columns = [str(column) for column in model_metadata["feature_columns"]]
    key_columns = {"trade_date", "stock_code"}
    # Output-only columns such as pred_prob and historical label/yield helpers
    # are not required from production_factor_parts for incremental scoring.
    optional_factor_columns = set(output_columns) - set(feature_columns) - key_columns
    factor_columns = sorted(set(output_columns + feature_columns + list(key_columns)))

    incremental_dates = [
        str(date)
        for date in sorted(set(pd.Series(pd.date_range(base_max, args.score_end)).dt.strftime("%Y%m%d")))
        if str(date) > base_max and str(date) <= args.score_end
    ]
    incremental = pd.DataFrame(columns=output_columns)
    if incremental_dates:
        factors = read_registered_factor_dates(
            data_dir,
            incremental_dates,
            factor_columns,
            optional_columns=optional_factor_columns | {args.label},
        )
        incremental = _predict_incremental(factors, metadata=model_metadata, label=args.label, output_columns=output_columns)

    combined = pd.concat([base[output_columns], incremental], ignore_index=True)
    combined["trade_date"] = combined["trade_date"].astype(str)
    combined = combined.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    duplicates = _dup_count(combined)
    if duplicates:
        raise RuntimeError(f"duplicate stock_code/trade_date keys in output: {duplicates}")
    if str(combined["trade_date"].min()) != args.score_start:
        raise RuntimeError(f"unexpected min_trade_date: {combined['trade_date'].min()}")
    if str(combined["trade_date"].max()) != args.score_end:
        raise RuntimeError(f"unexpected max_trade_date: {combined['trade_date'].max()}")

    _write_duckdb_table(db_path, args.output_table, combined)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest_dir = resolve_prediction_run_dir(data_dir, label=args.label, output_table=args.output_table, create=True)
    payload = _manifest_payload(
        label=args.label,
        candidate_id=args.candidate_id,
        output_table=args.output_table,
        base_table=args.base_table,
        frame=combined,
        model_metadata=model_metadata,
        incremental_dates=incremental_dates,
        generated_at=generated_at,
        output_db_path=db_path,
        manifest_dir=manifest_dir,
    )
    manifest_path = write_prediction_manifest(manifest_dir / "prediction_manifest.json", payload)
    status = {
        "status": "ok",
        "label": args.label,
        "candidate_id": args.candidate_id,
        "db_path": str(db_path),
        "output_table": args.output_table,
        "row_count": int(len(combined)),
        "trade_days": int(combined["trade_date"].nunique()),
        "stock_count": int(combined["stock_code"].nunique()),
        "min_trade_date": str(combined["trade_date"].min()),
        "max_trade_date": str(combined["trade_date"].max()),
        "duplicate_keys": duplicates,
        "base_max_trade_date": base_max,
        "incremental_score_dates": incremental_dates,
        "latest_date_counts": dict(list(_date_counts(combined).items())[-5:]),
        "manifest_path": str(manifest_path),
    }
    (manifest_dir / "build_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
