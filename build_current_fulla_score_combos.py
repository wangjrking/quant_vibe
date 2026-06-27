"""Build current-data full-A score ensemble tables.

This script intentionally reads only prediction tables rebuilt from the current
raw-data/light-factor chain. It does not read old stitched production tables.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


DATA_DIR = Path(r"D:\work\quant\quant_mcp\quant\data_file")
DB_PATH = DATA_DIR / "odb.db"
REPORT_DIR = DATA_DIR / "reports" / "current_full_a_search_20260615" / "score_combos"
COMBO_PREFIX = "stock_predict_data_current_fulla_combo_20260615"

SOURCES = {
    "c5_top10": "stock_predict_data_current_fulla_light_v1_executable_5d_open_top10_3y_fs120_class_depth3_lambda3",
    "c10_top10": "stock_predict_data_current_fulla_light_v1_executable_10d_open_top10_3y_fs120_class_depth3_lambda3",
    "r5": "stock_predict_data_current_fulla_light_v1_executable_5d_open_return_3y_fs120_colsample_bytree0_8_learning_rate0_01_max_depth3_n_estimators800_reg_lambda3_subsample0_8",
    "r10": "stock_predict_data_current_fulla_light_v1_executable_10d_open_return_3y_fs120_colsample_bytree0_8_learning_rate0_01_max_depth3_n_estimators800_reg_lambda3_subsample0_8",
    "c5_top05": "stock_predict_data_current_fulla_light_v1_executable_5d_open_top05_3y_fs120_class_depth3_lambda3",
}

COMBOS: dict[str, dict[str, float] | str] = {
    "rank_c5_70_c10_30": {"rank_c5_top10": 0.70, "rank_c10_top10": 0.30},
    "rank_c5_50_c10_50": {"rank_c5_top10": 0.50, "rank_c10_top10": 0.50},
    "rank_c5_80_r5_20": {"rank_c5_top10": 0.80, "rank_r5": 0.20},
    "rank_c5_70_r5_20_c10_10": {"rank_c5_top10": 0.70, "rank_r5": 0.20, "rank_c10_top10": 0.10},
    "rank_c5_60_r5_20_r10_20": {"rank_c5_top10": 0.60, "rank_r5": 0.20, "rank_r10": 0.20},
    "rank_c5_60_c10_20_r5_20": {"rank_c5_top10": 0.60, "rank_c10_top10": 0.20, "rank_r5": 0.20},
    "rank_c5_50_c10_20_r5_20_r10_10": {
        "rank_c5_top10": 0.50,
        "rank_c10_top10": 0.20,
        "rank_r5": 0.20,
        "rank_r10": 0.10,
    },
    "rank_max_c5_c10_r5": "max_c5_c10_r5",
    "rank_min_c5_c10": "min_c5_c10",
    "rank_c5_top10_top05_avg": {"rank_c5_top10": 0.70, "rank_c5_top05": 0.30},
}

BASE_COLUMNS = [
    "trade_date",
    "stock_code",
    "name",
    "post_open",
    "post2_open",
    "post4_open",
    "post6_open",
    "post12_open",
    "amount",
    "turnover_rate",
    "total_mv",
    "atr_qfq",
    "close",
    "limit_times",
    "st_type",
]


def _read_source(conn: sqlite3.Connection, alias: str, table: str, include_base: bool) -> pd.DataFrame:
    cols = ["trade_date", "stock_code", "pred_prob"]
    if include_base:
        cols = BASE_COLUMNS + ["pred_prob"]
    select_cols = ", ".join(cols)
    frame = pd.read_sql(f'select {select_cols} from "{table}"', conn)
    pred_col = f"pred_{alias}"
    rank_col = f"rank_{alias}"
    frame = frame.rename(columns={"pred_prob": pred_col})
    frame[pred_col] = pd.to_numeric(frame[pred_col], errors="coerce")
    frame[rank_col] = frame.groupby("trade_date")[pred_col].rank(pct=True)
    return frame


def build_combo_tables() -> list[dict[str, str]]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        source_items = list(SOURCES.items())
        merged = _read_source(conn, source_items[0][0], source_items[0][1], include_base=True)
        for alias, table in source_items[1:]:
            part = _read_source(conn, alias, table, include_base=False)
            merged = merged.merge(part, on=["trade_date", "stock_code"], how="inner")

        manifests: list[dict[str, str]] = []
        rank_cols = [f"rank_{alias}" for alias in SOURCES]
        pred_cols = [f"pred_{alias}" for alias in SOURCES]
        for name, spec in COMBOS.items():
            combo = merged.copy()
            if spec == "max_c5_c10_r5":
                combo["pred_prob"] = combo[["rank_c5_top10", "rank_c10_top10", "rank_r5"]].max(axis=1)
            elif spec == "min_c5_c10":
                combo["pred_prob"] = combo[["rank_c5_top10", "rank_c10_top10"]].min(axis=1)
            else:
                total = sum(float(value) for value in spec.values())
                combo["pred_prob"] = sum(combo[col] * float(weight) for col, weight in spec.items()) / total

            output_table = f"{COMBO_PREFIX}_{name}"
            write_cols = BASE_COLUMNS + ["pred_prob"] + pred_cols + rank_cols
            combo[write_cols].to_sql(output_table, conn, if_exists="replace", index=False)
            manifests.append(
                {
                    "combo": name,
                    "table": output_table,
                    "spec": json.dumps(spec, ensure_ascii=False),
                    "rows": str(len(combo)),
                    "dates": f"{combo['trade_date'].min()}-{combo['trade_date'].max()}",
                    "stocks": str(combo["stock_code"].nunique()),
                }
            )

        pd.DataFrame(manifests).to_sql("model_grid_current_fulla_combo_manifest_20260615", conn, if_exists="replace", index=False)
        pd.DataFrame(manifests).to_csv(REPORT_DIR / "combo_manifest.csv", index=False, encoding="utf-8-sig")
        conn.commit()
        return manifests
    finally:
        conn.close()


def main() -> None:
    manifests = build_combo_tables()
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
