from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "old500_vs_latest_l4_selection"
WIDE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
OLD_SIGNAL = (
    ROOT
    / "quant"
    / "main"
    / "strategy_library"
    / "production"
    / "prod_repro500_p435_s96_p10d70_v20260702"
    / "signals"
    / "full_history_repro500_p435_s96_p10d70.csv"
)
LATEST_SIGNAL = (
    REPORT_DIR
    / "signals"
    / "w35_15_00_50_pctm1p75_gapm0p08to0p0_r1070_r160.csv"
)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(OLD_SIGNAL, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    latest = pd.read_csv(LATEST_SIGNAL, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    old["old_selected"] = 1
    latest["latest_selected"] = 1

    con = duckdb.connect(str(WIDE_DB), read_only=True)
    try:
        con.register("old_sig", old)
        con.register("latest_sig", latest)
        old_join = con.execute(
            """
            SELECT
                o.signal_date,
                o.buy_date,
                o.stock_code,
                o.name AS old_name,
                o.rank AS old_rank,
                o.pred_prob AS old_pred_prob,
                o.pred_1d AS old_pred_1d,
                o.pred_3d AS old_pred_3d,
                o.pred_5d AS old_pred_5d,
                o.pred_10d AS old_pred_10d,
                o.pct_chg AS old_pct_chg,
                o.buy_open_gap_pct AS old_buy_open_gap_pct,
                w.name AS latest_name,
                w.pred_prob AS latest_pred_prob,
                w.score_pct_rank AS latest_score_pct_rank,
                w.score_desc_rank AS latest_score_desc_rank,
                w.signal_pct_chg,
                w.buy_open_gap_raw_pct,
                w.signal_amount,
                w.signal_turnover_rate,
                w.signal_total_mv,
                w.signal_atr_qfq,
                w.ret_h1,
                CASE WHEN l.latest_selected = 1 THEN 1 ELSE 0 END AS selected_by_latest_rule
            FROM old_sig o
            LEFT JOIN active_l4_wide w
              ON w.signal_date = o.signal_date AND w.stock_code = o.stock_code
            LEFT JOIN latest_sig l
              ON l.signal_date = o.signal_date AND l.stock_code = o.stock_code
            ORDER BY o.signal_date, o.rank
            """
        ).fetchdf()

        same_days_top = con.execute(
            """
            WITH old_days AS (
                SELECT DISTINCT signal_date FROM old_sig
            ),
            latest_ranked AS (
                SELECT
                    l.signal_date,
                    l.stock_code,
                    l.name,
                    l.rank,
                    l.pred_prob,
                    l.rank_1d,
                    l.rank_3d,
                    l.rank_5d,
                    l.rank_10d,
                    l.pct_chg,
                    l.buy_open_gap_raw_pct,
                    l.buy_open_gap_pct,
                    l.target_pct
                FROM latest_sig l
                JOIN old_days d USING(signal_date)
            )
            SELECT * FROM latest_ranked ORDER BY signal_date, rank
            """
        ).fetchdf()
    finally:
        con.close()

    old_join.to_csv(OUT_DIR / "old500_signals_latest_l4_join.csv", index=False, encoding="utf-8-sig")
    same_days_top.to_csv(OUT_DIR / "latest_rule_top_on_old500_days.csv", index=False, encoding="utf-8-sig")

    matched = old_join[old_join["latest_pred_prob"].notna()].copy()
    summary = {
        "old_rows": int(len(old)),
        "old_signal_days": int(old["signal_date"].nunique()),
        "old_max_signal_date": str(old["signal_date"].max()),
        "joined_to_latest_wide_rows": int(len(matched)),
        "missing_from_latest_wide_rows": int(old_join["latest_pred_prob"].isna().sum()),
        "old_rows_selected_by_latest_rule": int(old_join["selected_by_latest_rule"].fillna(0).sum()),
        "old_rows_selected_by_latest_rule_ratio": float(old_join["selected_by_latest_rule"].fillna(0).mean()),
        "latest_score_rank_summary_for_old": {
            "mean": float(matched["latest_score_pct_rank"].mean()) if len(matched) else None,
            "median": float(matched["latest_score_pct_rank"].median()) if len(matched) else None,
            "p90": float(matched["latest_score_pct_rank"].quantile(0.9)) if len(matched) else None,
            "min": float(matched["latest_score_pct_rank"].min()) if len(matched) else None,
            "max": float(matched["latest_score_pct_rank"].max()) if len(matched) else None,
        },
        "latest_desc_rank_for_old": {
            "median": float(matched["latest_score_desc_rank"].median()) if len(matched) else None,
            "p25": float(matched["latest_score_desc_rank"].quantile(0.25)) if len(matched) else None,
            "p75": float(matched["latest_score_desc_rank"].quantile(0.75)) if len(matched) else None,
        },
        "same_old_days_latest_rule_rows": int(len(same_days_top)),
        "same_old_days_latest_rule_days": int(same_days_top["signal_date"].nunique()) if len(same_days_top) else 0,
    }
    (OUT_DIR / "selection_drift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
