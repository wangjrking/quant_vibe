from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from ai_module import get_factor_data
from mlp_model_module import build_prediction_frame, predict_with_mlp


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a PyTorch MLP and write a prediction table.")
    parser.add_argument("--label", default="10d_yield_rate")
    parser.add_argument("--train-start", default="20100101")
    parser.add_argument("--test-start", default="20240604")
    parser.add_argument("--data-dir", default="../data_file")
    parser.add_argument("--stock-pool")
    parser.add_argument("--output-table", default="stock_predict_data_10d_yield_rate_mlp")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    train_x, train_y, test_x, test_y, train_data, test_data = get_factor_data(
        args.train_start,
        args.test_start,
        args.label,
        str(data_dir),
        args.stock_pool,
    )
    print(f"mlp_factor_split_done train={train_x.shape} test={test_x.shape}", flush=True)
    pred_y = predict_with_mlp(
        train_x,
        train_y,
        test_x,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
    )
    pred_data = build_prediction_frame(test_data, test_y, pred_y)
    with sqlite3.connect(data_dir / "odb.db") as conn:
        pred_data.to_sql(args.output_table, con=conn, if_exists="replace", index=False)
    print(
        f"mlp_pdb_update_done rows={pred_data.shape[0]} "
        f"max={pred_data['trade_date'].astype(str).max()} table={args.output_table}",
        flush=True,
    )


if __name__ == "__main__":
    main()
