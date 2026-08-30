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


REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


CASES = [
    {
        "name": "w50_00_00_50_gapm8p3_top3",
        "weights": {"1d": 0.50, "3d": 0.00, "5d": 0.00, "10d": 0.50},
        "topn": 3,
        "gap_low": -0.08,
        "gap_high": 0.03,
    },
    {
        "name": "w25_25_00_50_gapm8p3_top8",
        "weights": {"1d": 0.25, "3d": 0.25, "5d": 0.00, "10d": 0.50},
        "topn": 8,
        "gap_low": -0.08,
        "gap_high": 0.03,
    },
    {
        "name": "w75_00_25_00_nogap_top3",
        "weights": {"1d": 0.75, "3d": 0.00, "5d": 0.25, "10d": 0.00},
        "topn": 3,
        "gap_low": -0.20,
        "gap_high": 0.20,
    },
]


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
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE trade_calendar AS
            SELECT
                trade_date,
                lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
            FROM (
                SELECT DISTINCT trade_date
                FROM marketdb.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            ORDER BY trade_date
            """
        )
        table_names = {label: sources[label]["table"] for label in sources}
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE base AS
            WITH preds AS (
                SELECT
                    p10.trade_date,
                    p10.stock_code,
                    p1.pred_prob AS pred_1d,
                    p3.pred_prob AS pred_3d,
                    p5.pred_prob AS pred_5d,
                    p10.pred_prob AS pred_10d
                FROM l4_10d."{table_names['10d']}" p10
                JOIN l4_5d."{table_names['5d']}" p5 USING (trade_date, stock_code)
                JOIN l4_3d."{table_names['3d']}" p3 USING (trade_date, stock_code)
                JOIN l4_1d."{table_names['1d']}" p1 USING (trade_date, stock_code)
                WHERE p10.trade_date BETWEEN '20220606' AND '20260630'
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
            market AS (
                SELECT
                    trade_date,
                    stock_code,
                    name,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg,
                    close,
                    open,
                    ST_TYPE,
                    ST_TYPE_name
                FROM marketdb.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                r.*,
                m.name,
                m.amount,
                m.turnover_rate,
                m.total_mv,
                m.atr_qfq,
                m.pct_chg,
                (m.open / NULLIF(prev.close, 0) - 1) AS open_gap,
                cal.next_trade_date AS buy_date
            FROM ranked r
            JOIN market m USING (trade_date, stock_code)
            LEFT JOIN trade_calendar cal ON cal.trade_date = r.trade_date
            LEFT JOIN market prev ON prev.trade_date = cal.next_trade_date AND prev.stock_code = r.stock_code
            WHERE cal.next_trade_date IS NOT NULL
              AND r.stock_code NOT LIKE '%.BJ'
              AND coalesce(m.ST_TYPE, '') IN ('', '0')
              AND coalesce(m.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(m.name, '') NOT LIKE 'ST%'
              AND coalesce(m.name, '') NOT LIKE '*ST%'
            """
        )

        manifest_rows = []
        for case in CASES:
            w = case["weights"]
            score_expr = f"({w['1d']} * r1 + {w['3d']} * r3 + {w['5d']} * r5 + {w['10d']} * r10)"
            case_dir = REPORT_DIR / case["name"]
            case_dir.mkdir(parents=True, exist_ok=True)
            score_db = case_dir / "score.duckdb"
            signal_file = case_dir / "signals.csv"
            score_df = con.execute(
                f"""
                SELECT trade_date, stock_code, {score_expr}::DOUBLE AS pred_prob
                FROM base
                """
            ).fetchdf()
            with duckdb.connect(str(score_db)) as out_con:
                out_con.register("score_df", score_df)
                out_con.execute(
                    """
                    CREATE OR REPLACE TABLE score AS
                    SELECT trade_date, stock_code, pred_prob FROM score_df
                    """
                )
            df = con.execute(
                f"""
                WITH scored AS (
                    SELECT
                        *,
                        {score_expr} AS entry_score
                    FROM base
                    WHERE open_gap BETWEEN {float(case['gap_low'])} AND {float(case['gap_high'])}
                ),
                ranked AS (
                    SELECT
                        *,
                        row_number() OVER (PARTITION BY trade_date ORDER BY entry_score DESC, stock_code ASC) AS rn
                    FROM scored
                )
                SELECT
                    trade_date AS signal_date,
                    buy_date,
                    stock_code,
                    name,
                    rn AS rank,
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
                    pct_chg
                FROM ranked
                WHERE rn <= {int(case['topn'])}
                ORDER BY signal_date, rn
                """
            ).fetchdf()
            df.insert(2, "symbol", df["stock_code"].map(_symbol))
            df["target_pct"] = "0.20000"
            df["holding_days"] = 1
            df["max_holding_days"] = 2
            df["score_exit_entry_ratio"] = "0.98000"
            df["min_holding_days_before_score_exit"] = 1
            df["score_continue_entry_ratio"] = "0.99000"
            df["signal_stop_loss_pct"] = "0.05000"
            df["signal_take_profit_pct"] = "0.08000"
            df["strategy_variant"] = case["name"]
            df["filter_name"] = f"gap_{case['gap_low']}_{case['gap_high']}"
            df["entry_weight_name"] = case["name"].split("_gap")[0]
            df["dynamic_hold_name"] = "h1m2_e098_c099_min1"
            df["buy_day_market_available"] = True
            df["buy_day_hard_gate_complete"] = True
            df["buy_day_st_rejected"] = False
            df["buy_day_open_limit_up_rejected"] = False
            df["latest_market_date"] = "20260701"
            df.to_csv(signal_file, index=False, encoding="utf-8")
            counts = df.groupby("signal_date").size()
            manifest_rows.append(
                {
                    "case_name": case["name"],
                    "signal_file": str(signal_file),
                    "score_db": str(score_db),
                    "score_table": "score",
                    "rows": int(len(df)),
                    "days": int(counts.size),
                    "days_below_topn": int((counts < int(case["topn"])).sum()),
                    "case": case,
                }
            )
        pd.DataFrame(manifest_rows).to_csv(REPORT_DIR / "candidate_signal_manifest.csv", index=False, encoding="utf-8")
        (REPORT_DIR / "candidate_signal_manifest.json").write_text(
            json.dumps(manifest_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(pd.DataFrame(manifest_rows).to_string(index=False))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
