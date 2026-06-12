import sqlite3
import argparse
from pathlib import Path

from ai_module import get_factor_data, model_assess
from project_paths import resolve_data_dir


DATA_DIR = resolve_data_dir()
LABEL = "10d_yield_rate"
TRAIN_START = "20100101"
TEST_START = "20260101"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and write prediction table.")
    parser.add_argument("--label", default=LABEL)
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--test-start", default=TEST_START)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-table")
    parser.add_argument("--stock-pool")
    parser.add_argument("--model-backend", default="xgb", choices=["xgb", "mlp", "fttransformer"])
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    print("pdb_update_start", flush=True)
    train_x, train_y, test_x, test_y, train_data, test_data = get_factor_data(
        args.train_start,
        args.test_start,
        args.label,
        str(data_dir),
        args.stock_pool,
    )
    if args.model_backend != "xgb":
        raise NotImplementedError(f"{args.model_backend} training is provided by dedicated experiment scripts.")
    print(
        f"factor_split_done train={train_x.shape} test={test_x.shape} "
        f"test_max={test_data['trade_date'].astype(str).max()}",
        flush=True,
    )
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
        output_table = args.output_table or f"stock_predict_data_{args.label}"
        pred_data.to_sql(output_table, con=conn, if_exists="replace", index=False)
    print(
        f"pdb_update_done rows={pred_data.shape[0]} "
        f"max={pred_data['trade_date'].astype(str).max()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
