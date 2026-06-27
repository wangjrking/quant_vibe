import sqlite3
import argparse
from pathlib import Path

from ai_module import get_factor_data, model_assess
from model_asset_route import (
    MODEL_FEATURE_MODE_LEGACY,
    MODEL_FEATURE_MODE_SPLIT,
    MODEL_PREDICTION_MODE_INDEPENDENT,
    MODEL_PREDICTION_MODE_LEGACY,
    require_legacy_model_asset_chain_opt_in,
    resolve_legacy_prediction_db_path,
    resolve_model_prediction_db_path,
    resolve_prediction_run_dir,
    use_legacy_prediction_db,
    write_prediction_manifest,
)
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
    parser.add_argument(
        "--feature-source",
        default=None,
        choices=[MODEL_FEATURE_MODE_SPLIT, MODEL_FEATURE_MODE_LEGACY],
    )
    parser.add_argument(
        "--prediction-output-mode",
        default=None,
        choices=[MODEL_PREDICTION_MODE_INDEPENDENT, MODEL_PREDICTION_MODE_LEGACY],
    )
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
        feature_source=args.feature_source,
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
    output_table = args.output_table or f"stock_predict_data_{args.label}"
    if use_legacy_prediction_db(args.prediction_output_mode):
        require_legacy_model_asset_chain_opt_in(reason="legacy odb xgb prediction output")
        db_path = resolve_legacy_prediction_db_path(data_dir)
        prediction_mode = MODEL_PREDICTION_MODE_LEGACY
    else:
        db_path = resolve_model_prediction_db_path(data_dir, create_parent=True)
        prediction_mode = MODEL_PREDICTION_MODE_INDEPENDENT
    with sqlite3.connect(db_path) as conn:
        pred_data.to_sql(output_table, con=conn, if_exists="replace", index=False)
    run_dir = resolve_prediction_run_dir(data_dir, label=args.label, output_table=output_table, create=True)
    write_prediction_manifest(
        run_dir / "prediction_manifest.json",
        {
            "label": args.label,
            "feature_source": args.feature_source or MODEL_FEATURE_MODE_SPLIT,
            "prediction_mode": prediction_mode,
            "prediction_db_path": str(db_path),
            "prediction_table": output_table,
            "row_count": int(pred_data.shape[0]),
        },
    )
    print(
        f"pdb_update_done rows={pred_data.shape[0]} "
        f"max={pred_data['trade_date'].astype(str).max()} db={db_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
