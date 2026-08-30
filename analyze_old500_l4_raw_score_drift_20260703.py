from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
sys.path.insert(0, str(MAIN))

from prediction_manifest import load_prediction_source_manifest  # noqa: E402


REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "old500_l4_raw_score_drift"
OLD_SIGNAL = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "qfq_rerank_refill_candidates"
    / "reconstruct_500_from_p44_plus_1dge_m002"
    / "signals.csv"
)

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(OLD_SIGNAL, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect()
    try:
        con.register("old_sig", old)
        sources = {
            label: load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)
            for label, path in MANIFESTS.items()
        }
        for label, source in sources.items():
            con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
        tables = {label: sources[label]["table"] for label in sources}
        joined = con.execute(
            f"""
            WITH current_scores AS (
                SELECT
                    p1.trade_date AS signal_date,
                    p1.stock_code,
                    p1.pred_prob AS cur_1d,
                    p3.pred_prob AS cur_3d,
                    p5.pred_prob AS cur_5d,
                    p10.pred_prob AS cur_10d,
                    percent_rank() OVER (PARTITION BY p1.trade_date ORDER BY p1.pred_prob) AS cur_rank_1d,
                    percent_rank() OVER (PARTITION BY p3.trade_date ORDER BY p3.pred_prob) AS cur_rank_3d,
                    percent_rank() OVER (PARTITION BY p5.trade_date ORDER BY p5.pred_prob) AS cur_rank_5d,
                    percent_rank() OVER (PARTITION BY p10.trade_date ORDER BY p10.pred_prob) AS cur_rank_10d
                FROM l4_1d."{tables['1d']}" p1
                JOIN l4_3d."{tables['3d']}" p3 USING (trade_date, stock_code)
                JOIN l4_5d."{tables['5d']}" p5 USING (trade_date, stock_code)
                JOIN l4_10d."{tables['10d']}" p10 USING (trade_date, stock_code)
                WHERE p1.trade_date IN (SELECT DISTINCT signal_date FROM old_sig)
            )
            SELECT
                o.signal_date,
                o.stock_code,
                o.name,
                o.rank AS old_rank,
                o.pred_prob AS old_blend,
                o.pred_1d AS old_1d,
                o.pred_3d AS old_3d,
                o.pred_5d AS old_5d,
                o.pred_10d AS old_10d,
                o.pct_chg,
                o.buy_open_gap_pct,
                c.cur_1d,
                c.cur_3d,
                c.cur_5d,
                c.cur_10d,
                c.cur_rank_1d,
                c.cur_rank_3d,
                c.cur_rank_5d,
                c.cur_rank_10d,
                0.25 * c.cur_rank_1d + 0.25 * c.cur_rank_3d + 0.50 * c.cur_rank_10d AS cur_w25_25_00_50
            FROM old_sig o
            LEFT JOIN current_scores c
              ON c.signal_date = o.signal_date AND c.stock_code = o.stock_code
            ORDER BY o.signal_date, o.rank
            """
        ).fetchdf()
    finally:
        con.close()

    joined.to_csv(OUT_DIR / "old500_current_l4_raw_score_join.csv", index=False, encoding="utf-8-sig")

    numeric = [
        "old_blend",
        "old_1d",
        "old_3d",
        "old_5d",
        "old_10d",
        "cur_1d",
        "cur_3d",
        "cur_5d",
        "cur_10d",
        "cur_rank_1d",
        "cur_rank_3d",
        "cur_rank_5d",
        "cur_rank_10d",
        "cur_w25_25_00_50",
    ]
    desc = joined[numeric].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    desc.to_csv(OUT_DIR / "score_distribution_compare.csv", encoding="utf-8-sig")

    corr_cols = [
        "old_blend",
        "old_1d",
        "old_3d",
        "old_5d",
        "old_10d",
        "cur_1d",
        "cur_3d",
        "cur_5d",
        "cur_10d",
        "cur_rank_1d",
        "cur_rank_3d",
        "cur_rank_5d",
        "cur_rank_10d",
        "cur_w25_25_00_50",
    ]
    corr = joined[corr_cols].corr(numeric_only=True)
    corr.to_csv(OUT_DIR / "score_correlation_compare.csv", encoding="utf-8-sig")

    summary = {
        "rows": int(len(joined)),
        "missing_current_l4_rows": int(joined["cur_1d"].isna().sum()),
        "old_1d_range": [float(joined["old_1d"].min()), float(joined["old_1d"].max())],
        "current_1d_range": [float(joined["cur_1d"].min()), float(joined["cur_1d"].max())],
        "old_10d_range": [float(joined["old_10d"].min()), float(joined["old_10d"].max())],
        "current_10d_range": [float(joined["cur_10d"].min()), float(joined["cur_10d"].max())],
        "median_old_blend": float(joined["old_blend"].median()),
        "median_current_w25_25_00_50": float(joined["cur_w25_25_00_50"].median()),
        "corr_old_blend_current_w25_25_00_50": float(corr.loc["old_blend", "cur_w25_25_00_50"]),
        "corr_old_10d_current_rank10d": float(corr.loc["old_10d", "cur_rank_10d"]),
        "corr_old_1d_current_rank1d": float(corr.loc["old_1d", "cur_rank_1d"]),
    }
    (OUT_DIR / "raw_score_drift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
