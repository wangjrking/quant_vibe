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
from stock_daily_data_route import resolve_stock_daily_duckdb_path  # noqa: E402


REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "active_l4_qfq_rebuild"
)
MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        label: load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)
        for label, path in MANIFESTS.items()
    }
    market_db = resolve_stock_daily_duckdb_path(require_exists=True)

    con = duckdb.connect()
    try:
        for label, source in sources.items():
            con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{Path(market_db).as_posix()}' AS marketdb (READ_ONLY)")
        tables = {label: sources[label]["table"] for label in sources}
        max_usable_signal_date = con.execute(
            """
            SELECT max(trade_date)
            FROM (
                SELECT
                    trade_date,
                    lead(trade_date) OVER (ORDER BY trade_date) AS buy_date
                FROM (SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA)
            )
            WHERE buy_date IS NOT NULL
            """
        ).fetchone()[0]

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE base AS
            WITH cal AS (
                SELECT
                    trade_date,
                    lead(trade_date) OVER (ORDER BY trade_date) AS buy_date
                FROM (
                    SELECT DISTINCT trade_date
                    FROM marketdb.STOCK_DAILY_DATA
                    WHERE trade_date BETWEEN '20220606' AND '20260701'
                )
            ),
            preds AS (
                SELECT
                    p10.trade_date,
                    p10.stock_code,
                    p1.pred_prob AS pred_1d,
                    p3.pred_prob AS pred_3d,
                    p5.pred_prob AS pred_5d,
                    p10.pred_prob AS pred_10d
                FROM l4_10d."{tables['10d']}" p10
                JOIN l4_5d."{tables['5d']}" p5 USING (trade_date, stock_code)
                JOIN l4_3d."{tables['3d']}" p3 USING (trade_date, stock_code)
                JOIN l4_1d."{tables['1d']}" p1 USING (trade_date, stock_code)
                WHERE p10.trade_date BETWEEN '20220606' AND '{max_usable_signal_date}'
            ),
            ranked AS (
                SELECT
                    *,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_1d) AS r1,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS r3,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS r5,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS r10
                FROM preds
            ),
            md AS (
                SELECT
                    trade_date,
                    stock_code,
                    name,
                    open_qfq,
                    close_qfq,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg,
                    ST_TYPE,
                    ST_TYPE_name
                FROM marketdb.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                r.*,
                (0.25 * r1 + 0.25 * r3 + 0.50 * r10)::DOUBLE AS entry_score,
                md.name,
                md.amount,
                md.turnover_rate,
                md.total_mv,
                md.atr_qfq,
                md.pct_chg,
                md.close_qfq AS signal_close_qfq,
                buy.open_qfq AS buy_open_qfq,
                buy.open_qfq / NULLIF(md.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                md.atr_qfq / NULLIF(md.close_qfq, 0) AS atr_pct_qfq,
                cal.buy_date,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY md.amount) AS amount_rank,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY md.total_mv) AS mv_rank,
                percent_rank() OVER (
                    PARTITION BY r.trade_date
                    ORDER BY md.atr_qfq / NULLIF(md.close_qfq, 0)
                ) AS atr_rank
            FROM ranked r
            JOIN cal ON cal.trade_date = r.trade_date
            JOIN md ON md.trade_date = r.trade_date AND md.stock_code = r.stock_code
            JOIN md buy ON buy.trade_date = cal.buy_date AND buy.stock_code = r.stock_code
            WHERE cal.buy_date IS NOT NULL
              AND r.stock_code NOT LIKE '%.BJ'
              AND coalesce(md.ST_TYPE, '') IN ('', '0')
              AND coalesce(md.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(md.name, '') NOT LIKE 'ST%'
              AND coalesce(md.name, '') NOT LIKE '*ST%'
              AND md.close_qfq IS NOT NULL
              AND buy.open_qfq IS NOT NULL
            """
        )

        score_df = con.execute(
            "SELECT trade_date, stock_code, entry_score AS pred_prob FROM base"
        ).fetchdf()
        score_db = REPORT_DIR / "score.duckdb"
        with duckdb.connect(str(score_db)) as out_con:
            out_con.register("score_df", score_df)
            out_con.execute(
                """
                CREATE OR REPLACE TABLE score AS
                SELECT trade_date, stock_code, pred_prob FROM score_df
                """
            )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE top8 AS
            WITH filtered AS (
                SELECT *
                FROM base
                WHERE buy_open_gap_qfq BETWEEN -0.08 AND 0.03
            ),
            ranked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY entry_score DESC, stock_code ASC
                    ) AS base_rank
                FROM filtered
            )
            SELECT *
            FROM ranked
            WHERE base_rank <= 8
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE selected AS
            WITH enriched AS (
                SELECT
                    *,
                    entry_score
                    + 0.035 * amount_rank
                    + 0.020 * mv_rank
                    - 0.040 * atr_rank AS rerank_score
                FROM top8
                WHERE abs(pct_chg) <= 12.0
                  AND amount >= 90000.0
                  AND total_mv >= 200000.0
                  AND (
                    (buy_open_gap_qfq BETWEEN -0.05 AND 0.00 AND atr_pct_qfq <= 0.12)
                    OR
                    (buy_open_gap_qfq BETWEEN -0.08 AND 0.03 AND atr_pct_qfq <= 0.18)
                  )
            ),
            picked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY rerank_score DESC, stock_code ASC
                    ) AS rank
                FROM enriched
                WHERE pct_chg <= -1.75
                  AND buy_open_gap_qfq <= 0.015
                  AND pred_10d >= 0.70
            )
            SELECT *
            FROM picked
            WHERE rank <= 3
            ORDER BY trade_date, rank
            """
        )
        df = con.execute(
            """
            SELECT
                trade_date AS signal_date,
                buy_date,
                stock_code,
                name,
                rank,
                entry_score AS pred_prob,
                entry_score,
                pred_1d,
                pred_3d,
                pred_5d,
                pred_10d,
                amount,
                turnover_rate,
                total_mv,
                atr_qfq,
                pct_chg,
                buy_open_gap_qfq
            FROM selected
            """
        ).fetchdf()
        df.insert(2, "symbol", df["stock_code"].map(_symbol))
        df["target_pct"] = "0.43500"
        df["holding_days"] = 1
        df["max_holding_days"] = 1
        df["score_exit_entry_ratio"] = "0.96000"
        df["min_holding_days_before_score_exit"] = 1
        df["score_continue_entry_ratio"] = "9.99000"
        df["signal_stop_loss_pct"] = "0.05000"
        df["signal_take_profit_pct"] = "0.08000"
        df["strategy_variant"] = "active_l4_qfq_best_rebuild_p435"
        df["filter_name"] = "active_l4_qfq_top8_refill"
        df["entry_weight_name"] = "w25_25_00_50"
        df["dynamic_hold_name"] = "h1m1_qfq_open_refill_rebuild"
        df["buy_day_market_available"] = True
        df["buy_day_hard_gate_complete"] = True
        df["buy_day_st_rejected"] = False
        df["buy_day_open_limit_up_rejected"] = False
        df["latest_market_date"] = "20260701"
        df["buy_open_gap_pct"] = df["buy_open_gap_qfq"] * 100.0
        df = df.drop(columns=["buy_open_gap_qfq"])

        signal_file = REPORT_DIR / "active_l4_qfq_best_rebuild_p435" / "signals.csv"
        signal_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(signal_file, index=False, encoding="utf-8")

        counts = df.groupby("signal_date").size()
        audit = {
            "source_manifests": {label: str(path) for label, path in MANIFESTS.items()},
            "loaded_sources": sources,
            "market_db": str(market_db),
            "max_usable_signal_date": max_usable_signal_date,
            "score_db": str(score_db),
            "score_table": "score",
            "signal_file": str(signal_file),
            "signal_rows": int(len(df)),
            "signal_days": int(counts.size),
            "days_below_top3": int((counts < 3).sum()),
            "avg_names_per_day": float(counts.mean()) if not counts.empty else 0.0,
            "score_date_range": con.execute("SELECT min(trade_date), max(trade_date), count(*) FROM base").fetchone(),
            "price_fields": {
                "open_filter": "buy.open_qfq / signal.close_qfq - 1",
                "atr_filter": "atr_qfq / close_qfq",
                "raw_ohlc_usage": "none for open-gap/ATR filters",
            },
            "hard_filters": {
                "exclude_bj": True,
                "exclude_st_by_ST_TYPE_and_ST_TYPE_name": True,
                "exclude_name_ST_prefix": True,
            },
        }
        (REPORT_DIR / "active_l4_qfq_best_rebuild_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
