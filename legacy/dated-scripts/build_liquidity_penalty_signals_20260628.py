from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
TUNE_MODULE_PATH = MAIN / "tune_current_prod_four_year_formal_20260628.py"
BASE_REPORT = DATA / "reports" / "strategy_agent_tune_four_year_formal_current_rules_20260628"
REPORT_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628"
ORDER_VALUE = 700000.0

ENTRY_CASES = [
    {"name": "w80_15_05_amt90_mv20", "w10d": 0.80, "w5d": 0.15, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
    {"name": "w75_20_05_amt90_mv20", "w10d": 0.75, "w5d": 0.20, "w3d": 0.05, "amount_min": 90000.0, "total_mv_min": 200000.0},
]
BETAS = [0.03, 0.06, 0.10, 0.15]
EXEC_NAME = "base_h2m3_c097_e097_ddtight"


def _load_tune_module():
    spec = importlib.util.spec_from_file_location("tune_mod", TUNE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TUNE_MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _case_name(entry: dict[str, Any], beta: float) -> str:
    btxt = str(beta).replace(".", "p")
    return f"{entry['name']}_liqpen_b{btxt}__{EXEC_NAME}"


def main() -> None:
    mod = _load_tune_module()
    manifest, _, sources = mod._load_strategy(mod.STRATEGY_DIR)
    duckdb_path = Path(str(sources["10d"]["db_path"])).resolve()
    import duckdb

    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{duckdb_path.as_posix()}' AS prod (READ_ONLY)")
        for entry in ENTRY_CASES:
            for beta in BETAS:
                case_name = _case_name(entry, beta)
                out_file = REPORT_DIR / "signals" / f"{case_name}.csv"
                if out_file.exists():
                    print(str(out_file))
                    continue
                sql = f"""
                WITH market_dates AS (
                    SELECT DISTINCT trade_date FROM prod.STOCK_DAILY_DATA
                ),
                next_dates AS (
                    SELECT
                        trade_date AS signal_date,
                        lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                        max(trade_date) OVER () AS latest_market_date
                    FROM market_dates
                ),
                base AS (
                    SELECT
                        p10.trade_date,
                        p10.stock_code,
                        p3.pred_prob AS pred_3d,
                        p5.pred_prob AS pred_5d,
                        p10.pred_prob AS pred_10d
                    FROM prod."{sources['10d']['table']}" p10
                    INNER JOIN prod."{sources['5d']['table']}" p5
                        ON p10.trade_date = p5.trade_date AND p10.stock_code = p5.stock_code
                    INNER JOIN prod."{sources['3d']['table']}" p3
                        ON p10.trade_date = p3.trade_date AND p10.stock_code = p3.stock_code
                ),
                ranked AS (
                    SELECT
                        trade_date,
                        stock_code,
                        pred_3d,
                        pred_5d,
                        pred_10d,
                        percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS rank_3d,
                        percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS rank_5d,
                        percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS rank_10d,
                        {entry['w10d']} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d)
                      + {entry['w5d']} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d)
                      + {entry['w3d']} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS raw_score
                    FROM base
                ),
                scored AS (
                    SELECT
                        r.trade_date AS signal_date,
                        nd.buy_date,
                        nd.latest_market_date,
                        r.stock_code,
                        md.name,
                        md.amount,
                        md.turnover_rate,
                        md.total_mv,
                        md.atr_qfq,
                        r.pred_3d,
                        r.pred_5d,
                        r.pred_10d,
                        r.rank_3d,
                        r.rank_5d,
                        r.rank_10d,
                        r.raw_score,
                        sqrt({ORDER_VALUE} / greatest(1.0, try_cast(md.amount AS DOUBLE) * 1000.0)) AS liquidity_impact_est,
                        r.raw_score - {beta} * sqrt({ORDER_VALUE} / greatest(1.0, try_cast(md.amount AS DOUBLE) * 1000.0)) AS entry_score
                    FROM ranked r
                    LEFT JOIN prod.STOCK_DAILY_DATA md
                        ON r.trade_date = md.trade_date AND r.stock_code = md.stock_code
                    LEFT JOIN next_dates nd
                        ON r.trade_date = nd.signal_date
                    WHERE
                        NOT (r.stock_code LIKE '%.BJ' OR substr(r.stock_code, 1, 1) IN ('4', '8'))
                        AND NOT (
                            upper(coalesce(md.name, '')) LIKE 'ST%%'
                            OR upper(coalesce(md.name, '')) LIKE '*ST%%'
                            OR coalesce(md.ST_TYPE_name, '') LIKE '%风险%'
                            OR (try_cast(md.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(md.ST_TYPE AS DOUBLE) <> 0)
                        )
                        AND NOT (instr(coalesce(md.name, ''), '退市') > 0)
                        AND coalesce(try_cast(md.limit_times AS DOUBLE), 0.0) = 0.0
                        AND try_cast(md.amount AS DOUBLE) >= {entry['amount_min']}
                        AND try_cast(md.total_mv AS DOUBLE) >= {entry['total_mv_min']}
                ),
                with_buy_checks AS (
                    SELECT
                        s.*,
                        CASE
                            WHEN s.buy_date IS NULL THEN TRUE
                            WHEN bm.stock_code IS NULL THEN FALSE
                            WHEN upper(coalesce(bm.name, '')) LIKE 'ST%%' THEN FALSE
                            WHEN upper(coalesce(bm.name, '')) LIKE '*ST%%' THEN FALSE
                            WHEN coalesce(bm.ST_TYPE_name, '') LIKE '%风险%' THEN FALSE
                            WHEN try_cast(bm.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(bm.ST_TYPE AS DOUBLE) <> 0 THEN FALSE
                            WHEN coalesce(try_cast(bm.limit_times AS DOUBLE), 0.0) > 0.0 THEN FALSE
                            WHEN try_cast(bm.pre_close AS DOUBLE) IS NULL OR try_cast(bm.open AS DOUBLE) IS NULL THEN FALSE
                            WHEN try_cast(bm.pre_close AS DOUBLE) <= 0 OR try_cast(bm.open AS DOUBLE) <= 0 THEN FALSE
                            WHEN try_cast(bm.open AS DOUBLE) >= try_cast(bm.pre_close AS DOUBLE) * (
                                1.0 + CASE WHEN substr(s.stock_code, 1, 3) IN ('300', '301', '688') THEN 0.20 ELSE 0.10 END
                            ) * 0.995 THEN FALSE
                            ELSE TRUE
                        END AS buy_day_ok
                    FROM scored s
                    LEFT JOIN prod.STOCK_DAILY_DATA bm
                        ON s.buy_date = bm.trade_date AND s.stock_code = bm.stock_code
                ),
                selected AS (
                    SELECT *, row_number() OVER (PARTITION BY signal_date ORDER BY entry_score DESC) AS rn
                    FROM with_buy_checks
                    WHERE buy_day_ok
                )
                SELECT
                    signal_date,
                    coalesce(buy_date, '') AS buy_date,
                    CASE
                        WHEN stock_code LIKE '%.SH' THEN 'SHSE.' || substr(stock_code, 1, 6)
                        WHEN stock_code LIKE '%.SZ' THEN 'SZSE.' || substr(stock_code, 1, 6)
                        ELSE 'SZSE.' || substr(stock_code, 1, 6)
                    END AS symbol,
                    stock_code,
                    name,
                    1 AS rank,
                    entry_score AS pred_prob,
                    entry_score,
                    raw_score,
                    liquidity_impact_est,
                    pred_3d,
                    pred_5d,
                    pred_10d,
                    rank_3d,
                    rank_5d,
                    rank_10d,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    '0.90000' AS target_pct,
                    2 AS holding_days,
                    3 AS max_holding_days,
                    '0.97000' AS score_exit_entry_ratio,
                    1 AS min_holding_days_before_score_exit,
                    '0.97000' AS score_continue_entry_ratio,
                    '0.05000' AS signal_stop_loss_pct,
                    '0.07000' AS signal_take_profit_pct,
                    '{case_name}' AS strategy_variant,
                    'liquidity_penalty_signal_date' AS filter_name,
                    '{entry['name']}' AS entry_weight_name,
                    '{EXEC_NAME}' AS dynamic_hold_name,
                    TRUE AS buy_day_market_available,
                    TRUE AS buy_day_hard_gate_complete,
                    FALSE AS buy_day_st_rejected,
                    FALSE AS buy_day_open_limit_up_rejected,
                    latest_market_date
                FROM selected
                WHERE rn = 1
                ORDER BY signal_date
                """
                rows = con.execute(sql).fetchall()
                cols = [desc[0] for desc in con.description]
                _write_rows(out_file, [dict(zip(cols, row)) for row in rows])
                print(str(out_file))
    finally:
        con.close()


if __name__ == "__main__":
    main()
