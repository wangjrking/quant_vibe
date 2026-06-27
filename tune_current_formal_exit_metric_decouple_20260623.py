from __future__ import annotations

import sqlite3

import tune_current_formal_3d5d10d_20260623 as base


OLD_FUSION_DB = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "latest_formal_5d10d_stclean_refill"
    / "fusion_5d10d_latest_20260622.db"
)

base.REPORT_DIR = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "current_formal_exit_metric_decouple"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_exit_metric_decouple.db"
base.SCORE_DB = base.REPORT_DIR / "scores_exit_metric_decouple.db"


def _cfg(name: str, *, exit_metric: str, exit_ratio: str = "0.94", exit_rank: str | None = "0.94") -> dict:
    env = {
        "GM_OPEN_DAILY_SCORE_EXIT": "1",
        "GM_SCORE_EXIT_ENTRY_RATIO": str(exit_ratio),
        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
    }
    if exit_rank is not None:
        env["GM_SCORE_EXIT_RANK"] = str(exit_rank)
    return {
        "name": name,
        "w10": 0.80,
        "w5": 0.20,
        "w3": 0.00,
        "top_k": 4,
        "target_pct_each": 0.3305,
        "cap": 0.50,
        "holding_days": 5,
        "exit_metric": exit_metric,
        "env": env,
    }


base.VARIANTS = [
    _cfg("exit_fusion094", exit_metric="fusion", exit_ratio="0.94", exit_rank="0.94"),
    _cfg("exit_rank10d094", exit_metric="rank10d", exit_ratio="0.94", exit_rank="0.94"),
    _cfg("exit_rank10d093", exit_metric="rank10d", exit_ratio="0.93", exit_rank="0.94"),
    _cfg("exit_rank10d_norank", exit_metric="rank10d", exit_ratio="0.94", exit_rank=None),
    _cfg("exit_rank5d094", exit_metric="rank5d", exit_ratio="0.94", exit_rank="0.94"),
    _cfg("exit_rank5d090", exit_metric="rank5d", exit_ratio="0.90", exit_rank="0.90"),
    _cfg("exit_raw10d094", exit_metric="pred10d", exit_ratio="0.94", exit_rank=None),
    _cfg("exit_raw5d094", exit_metric="pred5d", exit_ratio="0.94", exit_rank=None),
    _cfg("exit_mix9010_094", exit_metric="mix9010", exit_ratio="0.94", exit_rank="0.94"),
    _cfg("exit_minrank094", exit_metric="minrank", exit_ratio="0.94", exit_rank="0.94"),
]


def _build_fusion_db() -> None:
    base.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if base.FUSION_DB.exists():
        return
    conn = sqlite3.connect(base.FUSION_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS old", (str(OLD_FUSION_DB),))
        conn.executescript(
            """
            CREATE TABLE fusion_rank_base AS
            SELECT
                trade_date,
                stock_code,
                0.0 AS pred_3d,
                pred_5d,
                pred_10d,
                0.0 AS rank_3d,
                rank_5d,
                rank_10d,
                name,
                pre_close,
                open,
                close,
                amount,
                turnover_rate,
                total_mv,
                atr_qfq,
                limit_times,
                st_type,
                st_type_name
            FROM old.fusion_rank_base;
            CREATE INDEX idx_fusion_trade_stock ON fusion_rank_base(trade_date, stock_code);
            CREATE INDEX idx_fusion_trade ON fusion_rank_base(trade_date);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _exit_expr(metric: str) -> str:
    return {
        "fusion": "(rank_10d * 0.8 + rank_5d * 0.2)",
        "rank10d": "rank_10d",
        "rank5d": "rank_5d",
        "pred10d": "pred_10d",
        "pred5d": "pred_5d",
        "mix9010": "(rank_10d * 0.9 + rank_5d * 0.1)",
        "minrank": "CASE WHEN rank_10d < rank_5d THEN rank_10d ELSE rank_5d END",
    }[metric]


def _build_score_table(cfg: dict) -> str:
    table = base._score_table(cfg)
    expr = _exit_expr(str(cfg["exit_metric"]))
    base.SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(base.SCORE_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS fusion", (str(base.FUSION_DB),))
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS "{table}";
            CREATE TABLE "{table}" AS
            SELECT trade_date, stock_code, {expr} AS pred_prob
            FROM fusion.fusion_rank_base
            WHERE trade_date >= '{base.START_DATE}' AND trade_date <= '{base.END_DATE}';
            CREATE INDEX idx_{table}_trade_stock ON "{table}"(trade_date, stock_code);
            CREATE INDEX idx_{table}_trade_pred ON "{table}"(trade_date, pred_prob DESC);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return table


def _build_signals(cfg: dict) -> list[dict]:
    market = base._market_rows()
    next_date = base._date_map()
    rank_expr = base._expr(cfg)
    exit_expr = _exit_expr(str(cfg["exit_metric"]))
    conn = sqlite3.connect(base.FUSION_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM (
                SELECT
                    trade_date, stock_code, name, pred_5d, pred_10d,
                    rank_5d, rank_10d, amount, turnover_rate, total_mv,
                    atr_qfq, close, limit_times, st_type, st_type_name,
                    {rank_expr} AS entry_score,
                    {exit_expr} AS exit_score,
                    ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY {rank_expr} DESC, stock_code) AS rn
                FROM fusion_rank_base
                WHERE trade_date >= ? AND trade_date <= ?
                  AND stock_code NOT LIKE '%.BJ'
                  AND COALESCE(name, '') NOT LIKE 'ST%'
                  AND COALESCE(name, '') NOT LIKE '*ST%'
                  AND COALESCE(name, '') NOT LIKE '%閫€%'
                  AND (st_type IS NULL OR st_type = '' OR st_type = 'None' OR UPPER(CAST(st_type AS TEXT)) IN ('0', '0.0', 'FALSE', 'NONE', 'NAN'))
                  AND COALESCE(st_type_name, '') NOT LIKE '%椋庨櫓%'
                  AND (limit_times IS NULL OR limit_times = '' OR limit_times = 'None' OR CAST(limit_times AS REAL) = 0)
                  AND rank_5d IS NOT NULL AND rank_10d IS NOT NULL
                  AND close IS NOT NULL
            )
            WHERE rn <= 4
            ORDER BY trade_date, rn
            """,
            (base.START_DATE, base.END_DATE),
        ).fetchall()
    finally:
        conn.close()
    signals = []
    for row in rows:
        item = dict(row)
        signal_date = str(item["trade_date"])
        buy_date = next_date.get(signal_date)
        if not buy_date:
            continue
        buy_market = market.get(buy_date, {}).get(str(item["stock_code"]))
        if base._is_st_like(buy_market) or base._is_limit_buy(buy_market):
            continue
        signals.append(
            {
                "signal_date": signal_date,
                "buy_date": buy_date,
                "symbol": base.to_gm_symbol(str(item["stock_code"])),
                "stock_code": item["stock_code"],
                "name": item.get("name"),
                "rank": int(item["rn"]),
                "pred_prob": item["exit_score"],
                "entry_score": item["entry_score"],
                "pred_5d": item.get("pred_5d"),
                "pred_10d": item.get("pred_10d"),
                "rank_5d": item.get("rank_5d"),
                "rank_10d": item.get("rank_10d"),
                "target_pct": "0.33050",
                "holding_days": 5,
                "max_holding_days": 5,
            }
        )
    return signals


base._build_fusion_db = _build_fusion_db
base._build_score_table = _build_score_table
base._build_signals = _build_signals


if __name__ == "__main__":
    raise SystemExit(base.main())
