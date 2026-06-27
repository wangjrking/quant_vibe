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
    / "current_formal_top1keep_others_exit"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_top1keep_others_exit.db"
base.SCORE_DB = base.REPORT_DIR / "scores_top1keep_others_exit.db"


def _cfg(
    name: str,
    *,
    others_exit_ratio: float,
    target_pct_each: float = 0.3305,
    top1_exit_ratio: float = 0.91,
    top1_holding_days: int = 7,
) -> dict:
    return {
        "name": name,
        "w10": 0.80,
        "w5": 0.20,
        "w3": 0.00,
        "top_k": 4,
        "target_pct_each": target_pct_each,
        "cap": 0.50,
        "holding_days": 5,
        "top1_rule": {
            "score_exit_entry_ratio": top1_exit_ratio,
            "holding_days": top1_holding_days,
        },
        "env": {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": str(others_exit_ratio),
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
        },
    }


base.VARIANTS = [
    _cfg("others0935", others_exit_ratio=0.935),
    _cfg("others0940", others_exit_ratio=0.94),
    _cfg("others0945", others_exit_ratio=0.945),
    _cfg("others0950", others_exit_ratio=0.95),
    _cfg("others0930", others_exit_ratio=0.93),
    _cfg("others0925", others_exit_ratio=0.925),
    _cfg("others0942", others_exit_ratio=0.942),
    _cfg("others0938", others_exit_ratio=0.938),
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


def _build_score_table(cfg: dict) -> str:
    table = base._score_table(cfg)
    base.SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(base.SCORE_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS fusion", (str(base.FUSION_DB),))
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS "{table}";
            CREATE TABLE "{table}" AS
            SELECT trade_date, stock_code, rank_10d AS pred_prob
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
    top1_rule = cfg.get("top1_rule") or {}
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
        rank = int(item["rn"])
        signal = {
            "signal_date": signal_date,
            "buy_date": buy_date,
            "symbol": base.to_gm_symbol(str(item["stock_code"])),
            "stock_code": item["stock_code"],
            "name": item.get("name"),
            "rank": rank,
            "pred_prob": item["rank_10d"],
            "entry_score": item["entry_score"],
            "pred_5d": item.get("pred_5d"),
            "pred_10d": item.get("pred_10d"),
            "rank_5d": item.get("rank_5d"),
            "rank_10d": item.get("rank_10d"),
            "target_pct": f"{float(cfg['target_pct_each']):.5f}",
            "holding_days": 5,
            "max_holding_days": 7,
        }
        if rank == 1:
            for key, value in top1_rule.items():
                signal[key] = value
        signals.append(signal)
    return signals


base._build_fusion_db = _build_fusion_db
base._build_score_table = _build_score_table
base._build_signals = _build_signals


if __name__ == "__main__":
    raise SystemExit(base.main())
