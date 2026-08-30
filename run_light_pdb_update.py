from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from ai_module import model_assess
from light_factor_module import get_light_factor_data
from model_asset_route import (
    MODEL_PREDICTION_MODE_INDEPENDENT,
    require_legacy_model_asset_chain_opt_in,
    resolve_prediction_run_dir,
    resolve_model_prediction_root,
    use_legacy_prediction_db,
    write_prediction_manifest,
)


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _prediction_duckdb_path(data_dir: Path, output_table: str) -> Path:
    safe_table = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in output_table).strip("_")
    return resolve_model_prediction_root(data_dir, create=True) / f"{safe_table}.duckdb"


def _write_prediction_duckdb(pred_data, duckdb_path: Path, table_name: str) -> None:
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(duckdb_path), read_only=False) as conn:
        conn.register("_prediction_frame", pred_data)
        conn.execute(f"CREATE OR REPLACE TABLE {_quote_ident(table_name)} AS SELECT * FROM _prediction_frame")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train prediction table from lightweight raw-data factors.")
    parser.add_argument("--label", default="10d_yield_rate")
    parser.add_argument("--train-start", default="20100101")
    parser.add_argument("--test-start", default="20240604")
    parser.add_argument("--end")
    parser.add_argument("--data-dir", default="data_file")
    parser.add_argument("--features", default="data_file/selected_features_10d_yield_rate_formula_fixed_backup_20260606.json")
    parser.add_argument("--stock-pool")
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument(
        "--prediction-output-mode",
        default=None,
        choices=[MODEL_PREDICTION_MODE_INDEPENDENT],
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    train_x, train_y, test_x, test_y, train_data, test_data = get_light_factor_data(
        data_dir=data_dir,
        features_path=args.features,
        train_start=args.train_start,
        test_start=args.test_start,
        label=args.label,
        end=args.end,
        stock_pool_path=args.stock_pool,
        allow_missing_features=args.allow_missing_features,
    )
    print(f"light_factor_split_done train={train_x.shape} test={test_x.shape}", flush=True)
    pred_data = model_assess(
        train_x,
        train_y,
        test_x,
        test_y,
        train_data,
        test_data,
        "reg",
        str(data_dir),
        save_shap=False,
    )
    if use_legacy_prediction_db(args.prediction_output_mode):
        require_legacy_model_asset_chain_opt_in(reason="legacy odb light-factor prediction output")
        raise RuntimeError("legacy SQLite prediction output is disabled in the DuckDB-only architecture.")
    else:
        db_path = _prediction_duckdb_path(data_dir, args.output_table)
        prediction_mode = MODEL_PREDICTION_MODE_INDEPENDENT
    _write_prediction_duckdb(pred_data, db_path, args.output_table)
    run_dir = resolve_prediction_run_dir(data_dir, label=args.label, output_table=args.output_table, create=True)
    write_prediction_manifest(
        run_dir / "prediction_manifest.json",
        {
            "label": args.label,
            "prediction_mode": prediction_mode,
            "source_type": "duckdb_table",
            "prediction_duckdb_path": str(db_path),
            "prediction_table": args.output_table,
            "row_count": int(pred_data.shape[0]),
        },
    )
    print(
        f"light_pdb_update_done rows={pred_data.shape[0]} "
        f"max={pred_data['trade_date'].astype(str).max()} table={args.output_table} db={db_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
