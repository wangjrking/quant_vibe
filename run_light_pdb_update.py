from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from ai_module import model_assess
from light_factor_module import get_light_factor_data
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
        choices=[MODEL_PREDICTION_MODE_INDEPENDENT, MODEL_PREDICTION_MODE_LEGACY],
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
        db_path = resolve_legacy_prediction_db_path(data_dir)
        prediction_mode = MODEL_PREDICTION_MODE_LEGACY
    else:
        db_path = resolve_model_prediction_db_path(data_dir, create_parent=True)
        prediction_mode = MODEL_PREDICTION_MODE_INDEPENDENT
    with sqlite3.connect(db_path) as conn:
        pred_data.to_sql(args.output_table, con=conn, if_exists="replace", index=False)
    run_dir = resolve_prediction_run_dir(data_dir, label=args.label, output_table=args.output_table, create=True)
    write_prediction_manifest(
        run_dir / "prediction_manifest.json",
        {
            "label": args.label,
            "prediction_mode": prediction_mode,
            "prediction_db_path": str(db_path),
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
