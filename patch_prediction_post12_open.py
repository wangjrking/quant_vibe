import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Patch post12_open into a prediction table from factor parquet.")
    parser.add_argument("--data-dir", default="data_file")
    parser.add_argument("--table", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    with sqlite3.connect(data_dir / "odb.db") as conn:
        pred = pd.read_sql(f'SELECT * FROM "{args.table}"', conn)
        factors = pd.read_parquet(
            data_dir / "stock_factor_data.parquet",
            columns=["stock_code", "trade_date", "post12_open", "post22_open"],
        )
        pred["trade_date"] = pred["trade_date"].astype(str)
        factors["trade_date"] = factors["trade_date"].astype(str)
        for col in ["post12_open", "post22_open"]:
            if col in pred.columns:
                pred = pred.drop(columns=[col])
        merged = pred.merge(factors, on=["stock_code", "trade_date"], how="left")
        merged.to_sql(args.table, con=conn, if_exists="replace", index=False)
    print(f"patched {args.table} rows={len(merged)} post12_non_null={merged['post12_open'].notna().sum()}")


if __name__ == "__main__":
    main()
