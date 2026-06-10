from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from ai_module import model_assess
from light_factor_module import get_light_factor_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train prediction table from lightweight raw-data factors.")
    parser.add_argument("--label", default="10d_yield_rate")
    parser.add_argument("--train-start", default="20100101")
    parser.add_argument("--test-start", default="20240604")
    parser.add_argument("--end")
    parser.add_argument("--data-dir", default="../data_file")
    parser.add_argument("--features", default="../data_file/selected_features_10d_yield_rate_formula_fixed_backup_20260606.json")
    parser.add_argument("--stock-pool")
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--allow-missing-features", action="store_true")
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
    with sqlite3.connect(data_dir / "odb.db") as conn:
        pred_data.to_sql(args.output_table, con=conn, if_exists="replace", index=False)
    print(
        f"light_pdb_update_done rows={pred_data.shape[0]} "
        f"max={pred_data['trade_date'].astype(str).max()} table={args.output_table}",
        flush=True,
    )


if __name__ == "__main__":
    main()
