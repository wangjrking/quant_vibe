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
    / "current_formal_exit10d_joint_refine"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_exit10d_joint_refine.db"
base.SCORE_DB = base.REPORT_DIR / "scores_exit10d_joint_refine.db"


def _cfg(
    name: str,
    *,
    target_pct_each: float,
    exit_ratio: str,
    min_hold: int,
    holding_days: int = 5,
    max_holding_days: int | None = None,
    continue_ratio: str | None = None,
    exit_rank: str | None = None,
) -> dict:
    env = {
        "GM_OPEN_DAILY_SCORE_EXIT": "1",
        "GM_SCORE_EXIT_ENTRY_RATIO": str(exit_ratio),
        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(min_hold),
    }
    if max_holding_days is not None:
        env["GM_MAX_HOLDING_DAYS"] = str(max_holding_days)
    if continue_ratio is not None:
        env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(continue_ratio)
    if exit_rank is not None:
        env["GM_SCORE_EXIT_RANK"] = str(exit_rank)
    return {
        "name": name,
        "w10": 0.80,
        "w5": 0.20,
        "w3": 0.00,
        "top_k": 4,
        "target_pct_each": target_pct_each,
        "cap": 0.50,
        "holding_days": holding_days,
        "env": env,
    }


base.VARIANTS = [
    _cfg("e10_t0328_r094_mh2", target_pct_each=0.3280, exit_ratio="0.94", min_hold=2),
    _cfg("e10_t0330_r094_mh2", target_pct_each=0.3300, exit_ratio="0.94", min_hold=2),
    _cfg("e10_t0331_r094_mh2", target_pct_each=0.3310, exit_ratio="0.94", min_hold=2),
    _cfg("e10_t0332_r094_mh2", target_pct_each=0.3320, exit_ratio="0.94", min_hold=2),
    _cfg("e10_t0333_r094_mh2", target_pct_each=0.3330, exit_ratio="0.94", min_hold=2),
    _cfg("e10_t0334_r094_mh2", target_pct_each=0.3340, exit_ratio="0.94", min_hold=2),
    _cfg("e10_t0335_r094_mh2", target_pct_each=0.3350, exit_ratio="0.94", min_hold=2),
    _cfg("e10_t0332_r0938_mh2", target_pct_each=0.3320, exit_ratio="0.938", min_hold=2),
    _cfg("e10_t0332_r0942_mh2", target_pct_each=0.3320, exit_ratio="0.942", min_hold=2),
    _cfg("e10_t0333_r0938_mh2", target_pct_each=0.3330, exit_ratio="0.938", min_hold=2),
    _cfg("e10_t0333_r0942_mh2", target_pct_each=0.3330, exit_ratio="0.942", min_hold=2),
    _cfg("e10_t0334_r0942_mh2", target_pct_each=0.3340, exit_ratio="0.942", min_hold=2),
    _cfg("e10_t0332_r094_mh2_h5x7_c099", target_pct_each=0.3320, exit_ratio="0.94", min_hold=2, holding_days=5, max_holding_days=7, continue_ratio="0.99"),
    _cfg("e10_t0332_r094_mh2_h5x8_c098", target_pct_each=0.3320, exit_ratio="0.94", min_hold=2, holding_days=5, max_holding_days=8, continue_ratio="0.98"),
    _cfg("e10_t0333_r094_mh2_h5x7_c099", target_pct_each=0.3330, exit_ratio="0.94", min_hold=2, holding_days=5, max_holding_days=7, continue_ratio="0.99"),
    _cfg("e10_t0333_r094_mh2_h5x8_c098", target_pct_each=0.3330, exit_ratio="0.94", min_hold=2, holding_days=5, max_holding_days=8, continue_ratio="0.98"),
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
        signal = {
            "signal_date": signal_date,
            "buy_date": buy_date,
            "symbol": base.to_gm_symbol(str(item["stock_code"])),
            "stock_code": item["stock_code"],
            "name": item.get("name"),
            "rank": int(item["rn"]),
            "pred_prob": item["rank_10d"],
            "entry_score": item["entry_score"],
            "pred_5d": item.get("pred_5d"),
            "pred_10d": item.get("pred_10d"),
            "rank_5d": item.get("rank_5d"),
            "rank_10d": item.get("rank_10d"),
            "target_pct": f"{float(cfg['target_pct_each']):.5f}",
            "holding_days": int(cfg["holding_days"]),
            "max_holding_days": int(base._to_float(cfg.get("env", {}).get("GM_MAX_HOLDING_DAYS"), cfg["holding_days"]) or cfg["holding_days"]),
        }
        continue_ratio = cfg.get("env", {}).get("GM_SCORE_CONTINUE_ENTRY_RATIO")
        if continue_ratio is not None:
            signal["score_continue_entry_ratio"] = continue_ratio
        signals.append(signal)
    return signals


base._build_fusion_db = _build_fusion_db
base._build_score_table = _build_score_table
base._build_signals = _build_signals


if __name__ == "__main__":
    raise SystemExit(base.main())
