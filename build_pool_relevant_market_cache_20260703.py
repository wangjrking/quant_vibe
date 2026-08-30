from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "merged_signal_pool_cache.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


def main() -> int:
    con = duckdb.connect(str(CACHE_DB))
    con.execute("PRAGMA threads=4")
    con.execute(f"ATTACH '{MARKET_DB.as_posix()}' AS marketdb (READ_ONLY)")
    con.execute(
        """
        CREATE OR REPLACE TABLE trade_calendar AS
        SELECT
            trade_date,
            lead(trade_date, 1) OVER (ORDER BY trade_date) AS d1,
            lead(trade_date, 2) OVER (ORDER BY trade_date) AS d2,
            lead(trade_date, 3) OVER (ORDER BY trade_date) AS d3,
            lead(trade_date, 5) OVER (ORDER BY trade_date) AS d5
        FROM (SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE needed_market_keys AS
        WITH p AS (
            SELECT DISTINCT buy_date, stock_code
            FROM pool
        ),
        expanded AS (
            SELECT p.stock_code, p.buy_date AS trade_date
            FROM p
            UNION
            SELECT p.stock_code, c.d1 AS trade_date FROM p JOIN trade_calendar c ON c.trade_date = p.buy_date WHERE c.d1 IS NOT NULL
            UNION
            SELECT p.stock_code, c.d2 AS trade_date FROM p JOIN trade_calendar c ON c.trade_date = p.buy_date WHERE c.d2 IS NOT NULL
            UNION
            SELECT p.stock_code, c.d3 AS trade_date FROM p JOIN trade_calendar c ON c.trade_date = p.buy_date WHERE c.d3 IS NOT NULL
            UNION
            SELECT p.stock_code, c.d5 AS trade_date FROM p JOIN trade_calendar c ON c.trade_date = p.buy_date WHERE c.d5 IS NOT NULL
        )
        SELECT DISTINCT stock_code, trade_date
        FROM expanded
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE relevant_market AS
        SELECT
            m.trade_date,
            m.stock_code,
            m.open,
            m.pre_close,
            m.amount,
            m.turnover_rate
        FROM marketdb.STOCK_DAILY_DATA m
        SEMI JOIN needed_market_keys k
          ON k.trade_date = m.trade_date
         AND k.stock_code = m.stock_code
        WHERE m.open IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE pool_enriched_small AS
        SELECT
            p.*,
            b.open / NULLIF(b.pre_close, 0) - 1 AS buy_day_open_gap,
            b.amount AS buy_amount,
            b.turnover_rate AS buy_turnover,
            s1.open / NULLIF(b.open, 0) - 1 AS ret_h1,
            s2.open / NULLIF(b.open, 0) - 1 AS ret_h2,
            s3.open / NULLIF(b.open, 0) - 1 AS ret_h3,
            s5.open / NULLIF(b.open, 0) - 1 AS ret_h5,
            percent_rank() OVER (PARTITION BY p.signal_date ORDER BY p.amount) AS amount_pctile,
            percent_rank() OVER (PARTITION BY p.signal_date ORDER BY p.total_mv) AS mv_pctile,
            percent_rank() OVER (PARTITION BY p.signal_date ORDER BY p.turnover_rate) AS turnover_pctile,
            percent_rank() OVER (PARTITION BY p.signal_date ORDER BY b.amount) AS buy_amount_pctile,
            percent_rank() OVER (PARTITION BY p.signal_date ORDER BY b.turnover_rate) AS buy_turnover_pctile
        FROM pool p
        JOIN trade_calendar c ON c.trade_date = p.buy_date
        JOIN relevant_market b ON b.trade_date = p.buy_date AND b.stock_code = p.stock_code
        JOIN relevant_market s1 ON s1.trade_date = c.d1 AND s1.stock_code = p.stock_code
        JOIN relevant_market s2 ON s2.trade_date = c.d2 AND s2.stock_code = p.stock_code
        JOIN relevant_market s3 ON s3.trade_date = c.d3 AND s3.stock_code = p.stock_code
        JOIN relevant_market s5 ON s5.trade_date = c.d5 AND s5.stock_code = p.stock_code
        """
    )
    summary = {
        "pool": con.execute("SELECT count(*) FROM pool").fetchone()[0],
        "needed_market_keys": con.execute("SELECT count(*) FROM needed_market_keys").fetchone()[0],
        "relevant_market": con.execute("SELECT count(*) FROM relevant_market").fetchone()[0],
        "pool_enriched_small": con.execute("SELECT count(*) FROM pool_enriched_small").fetchone()[0],
    }
    print(summary)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
