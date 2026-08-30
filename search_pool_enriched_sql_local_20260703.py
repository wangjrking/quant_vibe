from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "merged_signal_pool_cache.duckdb"
OUT_DIR = REPORT_DIR / "pool_enriched_sql_variants"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CACHE_DB))
    con.execute("PRAGMA threads=4")
    params = []
    cid = 0
    for pct_max in [-1.75, -3.0, -5.0, -7.0, -10.0]:
        for signal_gap_low, signal_gap_high in [(-8.0, 1.5), (-5.0, 1.5), (-3.0, 0.0), (-1.0, 0.0)]:
            for buy_gap_low, buy_gap_high in [(-0.08, 0.03), (-0.05, 0.015), (-0.03, 0.01), (-0.01, 0.01)]:
                for r10 in [0.70, 0.80, 0.90, 0.95]:
                    for r1 in [0.0, 0.60, 0.80, 0.90]:
                        for mode in ["blend", "liq", "turnlow", "buyliq"]:
                            for topn in [1, 2, 3]:
                                for hold_days in [1, 2, 3, 5]:
                                    cid += 1
                                    name = (
                                        f"psql_p{pct_max}_sg{signal_gap_low}_{signal_gap_high}"
                                        f"_bg{buy_gap_low}_{buy_gap_high}_r10{r10}_r1{r1}_{mode}_top{topn}_h{hold_days}"
                                    )
                                    name = name.replace("-", "m").replace(".", "p")
                                    params.append(
                                        (
                                            cid,
                                            name,
                                            pct_max,
                                            signal_gap_low,
                                            signal_gap_high,
                                            buy_gap_low,
                                            buy_gap_high,
                                            r10,
                                            r1,
                                            mode,
                                            topn,
                                            hold_days,
                                            min(0.99, 0.99 / topn),
                                        )
                                    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE param_grid(
            case_id INTEGER,
            case_name VARCHAR,
            pct_max DOUBLE,
            signal_gap_low DOUBLE,
            signal_gap_high DOUBLE,
            buy_gap_low DOUBLE,
            buy_gap_high DOUBLE,
            r10_min DOUBLE,
            r1_min DOUBLE,
            mode VARCHAR,
            topn INTEGER,
            hold_days INTEGER,
            target_pct DOUBLE
        )
        """
    )
    con.executemany("INSERT INTO param_grid VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", params)
    con.execute(
        """
        CREATE OR REPLACE TABLE pool_sql_selected AS
        WITH eligible AS (
            SELECT
                g.*,
                p.*,
                CASE g.mode
                    WHEN 'liq' THEN p.blend_score + 0.03 * p.amount_pctile + 0.02 * p.mv_pctile
                    WHEN 'turnlow' THEN p.blend_score - 0.04 * p.turnover_pctile + 0.02 * p.amount_pctile
                    WHEN 'buyliq' THEN p.blend_score + 0.03 * p.buy_amount_pctile - 0.02 * p.buy_turnover_pctile
                    ELSE p.blend_score
                END AS select_score
            FROM param_grid g
            JOIN pool_enriched_small p
              ON p.pct_chg <= g.pct_max
             AND p.buy_open_gap_raw_pct BETWEEN g.signal_gap_low AND g.signal_gap_high
             AND p.buy_day_open_gap BETWEEN g.buy_gap_low AND g.buy_gap_high
             AND p.rank_10d >= g.r10_min
             AND p.rank_1d >= g.r1_min
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY case_id, signal_date
                    ORDER BY select_score DESC, stock_code
                ) AS pick_rank
            FROM eligible
        )
        SELECT *
        FROM ranked
        WHERE pick_rank <= topn
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE pool_sql_local_summary AS
        WITH trade_ret AS (
            SELECT
                *,
                CASE hold_days
                    WHEN 1 THEN (1 + ret_h1) * (1 - 0.0015) / (1 + 0.0015) - 1
                    WHEN 2 THEN (1 + ret_h2) * (1 - 0.0015) / (1 + 0.0015) - 1
                    WHEN 3 THEN (1 + ret_h3) * (1 - 0.0015) / (1 + 0.0015) - 1
                    WHEN 5 THEN (1 + ret_h5) * (1 - 0.0015) / (1 + 0.0015) - 1
                END AS net_ret
            FROM pool_sql_selected
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
                any_value(case_name) AS case_name,
                signal_date,
                any_value(topn) AS topn,
                any_value(hold_days) AS hold_days,
                any_value(target_pct) AS target_pct,
                sum(net_ret * target_pct * CASE WHEN target_pct * names_in_day > 1 THEN 1.0 / (target_pct * names_in_day) ELSE 1.0 END) AS daily_ret,
                count(*) AS daily_names
            FROM sized
            GROUP BY case_id, signal_date
        ),
        equity AS (
            SELECT
                *,
                exp(sum(ln(greatest(0.0001, 1 + daily_ret))) OVER (
                    PARTITION BY case_id
                    ORDER BY signal_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                )) AS equity
            FROM daily
        ),
        dd AS (
            SELECT
                *,
                max(equity) OVER (
                    PARTITION BY case_id
                    ORDER BY signal_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS peak
            FROM equity
        ),
        hit AS (
            SELECT
                case_id,
                avg(CASE WHEN net_ret > 0 THEN 1.0 ELSE 0.0 END) AS win_ratio
            FROM trade_ret
            GROUP BY case_id
        )
        SELECT
            d.case_id,
            any_value(d.case_name) AS name,
            count(*) AS days,
            sum(d.daily_names) AS open_count,
            any_value(d.topn) AS topn,
            any_value(d.hold_days) AS hold_days,
            any_value(d.target_pct) AS target_pct,
            pow(exp(sum(ln(greatest(0.0001, 1 + d.daily_ret)))), 252.0 / count(*)) - 1 AS local_annual,
            avg(d.daily_ret) / nullif(stddev_samp(d.daily_ret), 0) * sqrt(252.0) AS local_sharpe,
            max(1 - dd.equity / nullif(dd.peak, 0)) AS local_max_drawdown,
            any_value(hit.win_ratio) AS win_ratio
        FROM daily d
        JOIN dd ON dd.case_id = d.case_id AND dd.signal_date = d.signal_date
        JOIN hit ON hit.case_id = d.case_id
        GROUP BY d.case_id
        HAVING count(*) >= 40 AND sum(d.daily_names) >= 60
        ORDER BY local_annual DESC, local_sharpe DESC
        """
    )
    summary_path = REPORT_DIR / "pool_enriched_sql_local_summary.csv"
    con.execute(f"COPY pool_sql_local_summary TO '{summary_path.as_posix()}' (HEADER, DELIMITER ',')")
    top_cases = con.execute(
        """
        SELECT case_id, name
        FROM pool_sql_local_summary
        WHERE local_max_drawdown <= 0.60
        ORDER BY local_annual DESC, local_sharpe DESC
        LIMIT 12
        """
    ).fetchall()
    manifest = []
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
                    name,
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
                    hold_days AS holding_days,
                    hold_days AS max_holding_days,
                    9.99 AS score_exit_entry_ratio,
                    1 AS min_holding_days_before_score_exit,
                    9.99 AS score_continue_entry_ratio,
                    NULL AS signal_stop_loss_pct,
                    NULL AS signal_take_profit_pct,
                    '{name}' AS strategy_variant,
                    'pool_enriched_sql' AS filter_name,
                    mode AS entry_weight_name,
                    'h' || CAST(hold_days AS VARCHAR) AS dynamic_hold_name,
                    true AS buy_day_market_available,
                    true AS buy_day_hard_gate_complete,
                    false AS buy_day_st_rejected,
                    false AS buy_day_open_limit_up_rejected,
                    '20260702' AS latest_market_date
                FROM pool_sql_selected
                WHERE case_id = {case_id}
                ORDER BY signal_date, pick_rank
            ) TO '{out_file.as_posix()}' (HEADER, DELIMITER ',')
            """
        )
        manifest.append({"case_id": case_id, "name": name, "signal_file": str(out_file)})
    manifest_path = OUT_DIR / "pool_enriched_sql_top12_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "name", "signal_file"])
        writer.writeheader()
        writer.writerows(manifest)
    (OUT_DIR / "pool_enriched_sql_top12_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(con.execute("SELECT * FROM pool_sql_local_summary ORDER BY local_annual DESC, local_sharpe DESC LIMIT 20").fetchdf().to_string(index=False))
    print(summary_path)
    print(manifest_path)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
