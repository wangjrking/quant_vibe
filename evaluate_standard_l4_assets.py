from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from evaluate_prediction_asset import evaluate_frame
from model_asset_route import resolve_model_prediction_db_path


DEFAULT_TABLES = {
    "executable_1d_open_return": "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_1d_open_return_score_20240604_20260618",
    "executable_3d_open_return": "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_3d_open_return_score_20240604_20260618",
    "executable_5d_open_return": "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_5d_open_return_score_20240604_20260618",
    "executable_10d_open_return": "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate current standard L4 model score assets.")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data_file"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--date-from", default="20240604")
    parser.add_argument("--date-to", default="20260618")
    parser.add_argument("--top-k", default="1,3,5,10,20,50")
    parser.add_argument("--quantiles", type=int, default=10)
    return parser.parse_args(argv)


def _split_top_k(raw: str) -> list[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _read_eval_frame(db_path: Path, table: str, label: str, date_from: str, date_to: str) -> pd.DataFrame:
    sql = (
        f"select trade_date, stock_code, pred_prob, {label} "
        f"from '{table}' "
        "where trade_date >= ? and trade_date <= ?"
    )
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(str(date_from), str(date_to)))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = resolve_model_prediction_db_path(data_dir)
    top_k = _split_top_k(args.top_k)

    rows = []
    for label, table in DEFAULT_TABLES.items():
        frame = _read_eval_frame(db_path, table, label, args.date_from, args.date_to)
        summary, daily = evaluate_frame(
            frame,
            score_col="pred_prob",
            label_col=label,
            top_k=top_k,
            quantiles=args.quantiles,
        )
        payload = {
            "source": f"{db_path.resolve()}::{table}",
            "table": table,
            "score_col": "pred_prob",
            "label_col": label,
            "date_filter_from": args.date_from,
            "date_filter_to": args.date_to,
            **summary,
        }
        _write_json(output_dir / f"{label}.eval.json", payload)
        daily.to_csv(output_dir / f"{label}.daily.csv", index=False, encoding="utf-8-sig")
        row = {
            "label": label,
            "table": table,
            "date_min": payload["date_min"],
            "date_max": payload["date_max"],
            "trade_days": payload["trade_days"],
            "total_rows": payload["total_rows"],
            "valid_rows": payload["valid_rows"],
            "valid_ratio": payload["valid_ratio"],
            "daily_pearson_ic_mean": payload["daily_pearson_ic_mean"],
            "daily_rank_ic_mean": payload["daily_rank_ic_mean"],
            "rank_ic_positive_ratio": payload["rank_ic_positive_ratio"],
            "top_decile_mean_return": payload["top_decile_mean_return"],
            "bottom_decile_mean_return": payload["bottom_decile_mean_return"],
            "top_minus_bottom_mean": payload["top_minus_bottom_mean"],
            "top_minus_bottom_positive_ratio": payload["top_minus_bottom_positive_ratio"],
        }
        for key, value in payload["top_k_mean_returns"].items():
            row[f"top{key}_mean_return"] = value
        rows.append(row)

    summary_frame = pd.DataFrame(rows)
    summary_path = output_dir / "standard_l4_model_eval_summary.csv"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "status": "ok",
                "db_path": str(db_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "summary_csv": str(summary_path.resolve()),
                "labels": list(DEFAULT_TABLES),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
