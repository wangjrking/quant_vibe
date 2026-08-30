from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_active_l4_prod_repro_grid_20260703"
)
SIGNAL_GLOB = (REPORT_DIR / "signals" / "*.csv").as_posix()
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
CACHE_DB = REPORT_DIR / "merged_signal_pool_cache.duckdb"
OUT_DIR = REPORT_DIR / "merged_signal_pool_cache_variants"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CACHE_DB))
    con.execute("PRAGMA threads=4")
    con.execute(f"ATTACH '{MARKET_DB.as_posix()}' AS marketdb (READ_ONLY)")
    existing_tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "raw_pool" not in existing_tables:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE raw_pool AS
            SELECT
                CAST(signal_date AS VARCHAR) AS signal_date,
                CAST(buy_date AS VARCHAR) AS buy_date,
                CAST(symbol AS VARCHAR) AS symbol,
                CAST(stock_code AS VARCHAR) AS stock_code,
                CAST(name AS VARCHAR) AS name,
                CAST(rank AS INTEGER) AS rank,
                CAST(pred_prob AS DOUBLE) AS pred_prob,
                CAST(entry_score AS DOUBLE) AS entry_score,
                CAST(pred_1d AS DOUBLE) AS pred_1d,
                CAST(pred_3d AS DOUBLE) AS pred_3d,
                CAST(pred_5d AS DOUBLE) AS pred_5d,
                CAST(pred_10d AS DOUBLE) AS pred_10d,
                CAST(rank_1d AS DOUBLE) AS rank_1d,
                CAST(rank_3d AS DOUBLE) AS rank_3d,
                CAST(rank_5d AS DOUBLE) AS rank_5d,
                CAST(rank_10d AS DOUBLE) AS rank_10d,
                CAST(amount AS DOUBLE) AS amount,
                CAST(turnover_rate AS DOUBLE) AS turnover_rate,
                CAST(total_mv AS DOUBLE) AS total_mv,
                CAST(atr_qfq AS DOUBLE) AS atr_qfq,
                CAST(pct_chg AS DOUBLE) AS pct_chg,
                CAST(buy_open_gap_raw_pct AS DOUBLE) AS buy_open_gap_raw_pct,
                filename AS source_file
            FROM read_csv(
                '{SIGNAL_GLOB}',
                header=true,
                union_by_name=true,
                filename=true,
                ignore_errors=true
            )
            WHERE stock_code IS NOT NULL
              AND stock_code NOT LIKE '%.BJ'
            """
        )
    if "pool" not in existing_tables:
        con.execute(
            """
            CREATE OR REPLACE TABLE pool AS
            WITH scored AS (
                SELECT
                    *,
                    0.15 * rank_1d + 0.35 * rank_3d + 0.50 * rank_10d AS blend_score,
                    row_number() OVER (
                        PARTITION BY signal_date, stock_code
                        ORDER BY 0.15 * rank_1d + 0.35 * rank_3d + 0.50 * rank_10d DESC, source_file
                    ) AS dup_rank
                FROM raw_pool
                WHERE rank_10d >= 0.70
                  AND pct_chg <= -1.0
                  AND buy_open_gap_raw_pct BETWEEN -8.0 AND 1.5
            )
            SELECT * EXCLUDE (dup_rank)
            FROM scored
            WHERE dup_rank = 1
            """
        )
    con.execute(
        """
        CREATE OR REPLACE TABLE pool_enriched AS
        WITH cal AS (
            SELECT
                trade_date,
                lead(trade_date, 1) OVER (ORDER BY trade_date) AS d1,
                lead(trade_date, 2) OVER (ORDER BY trade_date) AS d2,
                lead(trade_date, 3) OVER (ORDER BY trade_date) AS d3,
                lead(trade_date, 5) OVER (ORDER BY trade_date) AS d5
            FROM (SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA)
        )
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
        JOIN marketdb.STOCK_DAILY_DATA b ON b.trade_date = p.buy_date AND b.stock_code = p.stock_code
        JOIN cal ON cal.trade_date = p.buy_date
        JOIN marketdb.STOCK_DAILY_DATA s1 ON s1.trade_date = cal.d1 AND s1.stock_code = p.stock_code
        JOIN marketdb.STOCK_DAILY_DATA s2 ON s2.trade_date = cal.d2 AND s2.stock_code = p.stock_code
        JOIN marketdb.STOCK_DAILY_DATA s3 ON s3.trade_date = cal.d3 AND s3.stock_code = p.stock_code
        JOIN marketdb.STOCK_DAILY_DATA s5 ON s5.trade_date = cal.d5 AND s5.stock_code = p.stock_code
        WHERE b.open IS NOT NULL
          AND b.pre_close IS NOT NULL
        """
    )
    params = []
    idx = 0
    for pct_max in [-3.0, -5.0, -7.0]:
        for buy_gap_low, buy_gap_high in [(-0.05, 0.015), (-0.03, 0.01), (-0.01, 0.01)]:
            for rank_1d_min in [0.0, 0.8, 0.9]:
                for rank_10d_min in [0.8, 0.9, 0.95]:
                    for score_mode in ["blend", "liq", "turnlow", "buyliq"]:
                        for topn in [1, 2, 3]:
                            for hold in [1, 2, 3, 5]:
                                idx += 1
                                target_pct = min(0.99, 0.99 / topn)
                                params.append(
                                    (
                                        idx,
                                        f"mpool_p{pct_max}_bg{buy_gap_low}_{buy_gap_high}_r1{rank_1d_min}_r10{rank_10d_min}_{score_mode}_top{topn}_h{hold}"
                                        .replace("-", "m")
                                        .replace(".", "p"),
                                        pct_max,
                                        buy_gap_low,
                                        buy_gap_high,
                                        rank_1d_min,
                                        rank_10d_min,
                                        score_mode,
                                        topn,
                                        hold,
                                        target_pct,
                                    )
                                )
    con.execute("CREATE OR REPLACE TABLE param_grid(case_id INTEGER, name VARCHAR, pct_max DOUBLE, buy_gap_low DOUBLE, buy_gap_high DOUBLE, rank_1d_min DOUBLE, rank_10d_min DOUBLE, score_mode VARCHAR, topn INTEGER, hold INTEGER, target_pct DOUBLE)")
    con.executemany("INSERT INTO param_grid VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", params)
    con.execute(
        """
        CREATE OR REPLACE TABLE selected AS
        WITH eligible AS (
            SELECT
                g.*,
                p.*,
                CASE g.score_mode
                    WHEN 'liq' THEN p.blend_score + 0.03 * p.amount_pctile + 0.02 * p.mv_pctile
                    WHEN 'turnlow' THEN p.blend_score - 0.05 * p.turnover_pctile + 0.02 * p.amount_pctile
                    WHEN 'buyliq' THEN p.blend_score + 0.03 * p.buy_amount_pctile - 0.02 * p.buy_turnover_pctile
                    ELSE p.blend_score
                END AS select_score
            FROM param_grid g
            JOIN pool_enriched p
              ON p.pct_chg <= g.pct_max
             AND p.buy_day_open_gap BETWEEN g.buy_gap_low AND g.buy_gap_high
             AND p.rank_1d >= g.rank_1d_min
             AND p.rank_10d >= g.rank_10d_min
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (PARTITION BY case_id, signal_date ORDER BY select_score DESC, stock_code) AS pick_rank
            FROM eligible
        )
        SELECT *
        FROM ranked
        WHERE pick_rank <= topn
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE local_summary AS
        WITH trade_ret AS (
            SELECT
                *,
                CASE hold
                    WHEN 1 THEN (1 + ret_h1) * (1 - 0.0015) / (1 + 0.0015) - 1
                    WHEN 2 THEN (1 + ret_h2) * (1 - 0.0015) / (1 + 0.0015) - 1
                    WHEN 3 THEN (1 + ret_h3) * (1 - 0.0015) / (1 + 0.0015) - 1
                    WHEN 5 THEN (1 + ret_h5) * (1 - 0.0015) / (1 + 0.0015) - 1
                END AS net_ret
            FROM selected
        ),
        sized AS (
            SELECT
                *,
                count(*) OVER (PARTITION BY case_id, signal_date) AS names_in_day
            FROM trade_ret
        ),
        daily AS (
            SELECT
                case_id,
                name,
                signal_date,
                avg(topn) AS topn,
                avg(hold) AS hold,
                avg(target_pct) AS target_pct,
                sum(net_ret * target_pct * CASE WHEN target_pct * names_in_day > 1 THEN 1.0 / (target_pct * names_in_day) ELSE 1 END) AS daily_ret,
                count(*) AS names
            FROM sized
            GROUP BY case_id, name, signal_date
        ),
        equity AS (
            SELECT
                *,
                exp(sum(ln(greatest(0.0001, 1 + daily_ret))) OVER (PARTITION BY case_id ORDER BY signal_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) AS equity
            FROM daily
        ),
        dd AS (
            SELECT
                *,
                max(equity) OVER (PARTITION BY case_id ORDER BY signal_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS peak
            FROM equity
        )
        SELECT
            d.case_id,
            any_value(d.name) AS name,
            count(*) AS days,
            sum(d.names) AS open_count,
            any_value(d.topn) AS topn,
            any_value(d.hold) AS hold,
            any_value(d.target_pct) AS target_pct,
            pow(exp(sum(ln(greatest(0.0001, 1 + d.daily_ret)))), 252.0 / count(*)) - 1 AS local_annual,
            avg(d.daily_ret) / nullif(stddev_samp(d.daily_ret), 0) * sqrt(252.0) AS local_sharpe,
            max(1 - dd.equity / nullif(dd.peak, 0)) AS local_max_drawdown,
            avg(CASE WHEN tr.net_ret > 0 THEN 1.0 ELSE 0.0 END) AS win_ratio
        FROM daily d
        JOIN dd ON dd.case_id = d.case_id AND dd.signal_date = d.signal_date
        JOIN trade_ret tr ON tr.case_id = d.case_id AND tr.signal_date = d.signal_date
        GROUP BY d.case_id
        HAVING count(*) >= 40 AND sum(d.names) >= 60
        ORDER BY local_annual DESC, local_sharpe DESC
        """
    )
    summary_path = REPORT_DIR / "merged_signal_pool_cache_local_summary.csv"
    con.execute(f"COPY local_summary TO '{summary_path.as_posix()}' (HEADER, DELIMITER ',')")
    top_cases = con.execute("SELECT case_id, name FROM local_summary ORDER BY local_annual DESC, local_sharpe DESC LIMIT 10").fetchall()
    variant_manifest = []
    for case_id, name in top_cases:
        out_file = OUT_DIR / f"{name}.csv"
        con.execute(
            f"""
            COPY (
                SELECT
                    signal_date,
                    buy_date,
                    symbol,
                    stock_code,
                    name AS name,
                    pick_rank AS rank,
                    entry_score AS pred_prob,
                    entry_score,
                    pred_1d,
                    pred_3d,
                    pred_5d,
                    pred_10d,
                    rank_1d,
                    rank_3d,
                    rank_5d,
                    rank_10d,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg,
                    buy_open_gap_raw_pct,
                    target_pct,
                    hold AS holding_days,
                    hold AS max_holding_days,
                    9.99 AS score_exit_entry_ratio,
                    1 AS min_holding_days_before_score_exit,
                    9.99 AS score_continue_entry_ratio,
                    NULL AS signal_stop_loss_pct,
                    NULL AS signal_take_profit_pct,
                    '{name}' AS strategy_variant,
                    'merged_signal_pool_cache' AS filter_name,
                    score_mode AS entry_weight_name,
                    'h' || CAST(hold AS VARCHAR) AS dynamic_hold_name,
                    true AS buy_day_market_available,
                    true AS buy_day_hard_gate_complete,
                    false AS buy_day_st_rejected,
                    false AS buy_day_open_limit_up_rejected,
                    '20260702' AS latest_market_date
                FROM selected
                WHERE case_id = {case_id}
                ORDER BY signal_date, pick_rank
            ) TO '{out_file.as_posix()}' (HEADER, DELIMITER ',')
            """
        )
        variant_manifest.append({"case_id": case_id, "name": name, "signal_file": str(out_file)})
    manifest_path = OUT_DIR / "merged_signal_pool_cache_top10_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "name", "signal_file"])
        writer.writeheader()
        writer.writerows(variant_manifest)
    (OUT_DIR / "merged_signal_pool_cache_top10_manifest.json").write_text(
        json.dumps(variant_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = con.execute("SELECT * FROM local_summary ORDER BY local_annual DESC, local_sharpe DESC LIMIT 20").fetchdf()
    print(summary.to_string(index=False))
    print(summary_path)
    print(manifest_path)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
