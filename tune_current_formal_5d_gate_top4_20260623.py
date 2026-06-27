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
    / "current_formal_5d_gate_top4"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_5d_gate_top4.db"
base.SCORE_DB = base.REPORT_DIR / "scores_5d_gate_top4.db"


def _cfg(name: str, *, gate_5d: float, order_mode: str = "10d") -> dict:
    return {
        "name": name,
        "w10": 1.0,
        "w5": 0.0,
        "w3": 0.0,
        "top_k": 4,
        "target_pct_each": 0.3305,
        "cap": 0.50,
        "holding_days": 5,
        "gate_5d": gate_5d,
        "order_mode": order_mode,
        "env": {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.94",
            "GM_SCORE_EXIT_RANK": "0.94",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
        },
    }


base.VARIANTS = [
    _cfg("gate5d40_order10d", gate_5d=0.40, order_mode="10d"),
    _cfg("gate5d45_order10d", gate_5d=0.45, order_mode="10d"),
    _cfg("gate5d50_order10d", gate_5d=0.50, order_mode="10d"),
    _cfg("gate5d55_order10d", gate_5d=0.55, order_mode="10d"),
    _cfg("gate5d60_order10d", gate_5d=0.60, order_mode="10d"),
    _cfg("gate5d50_ordermix", gate_5d=0.50, order_mode="mix"),
    _cfg("gate5d55_ordermix", gate_5d=0.55, order_mode="mix"),
    _cfg("gate5d60_ordermix", gate_5d=0.60, order_mode="mix"),
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
            SELECT trade_date, stock_code, pred_10d AS pred_prob
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
    conn = sqlite3.connect(base.FUSION_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_date, stock_code, name, pred_5d, pred_10d, rank_5d, rank_10d,
                   amount, turnover_rate, total_mv, atr_qfq, close, limit_times, st_type, st_type_name
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
            ORDER BY trade_date, rank_10d DESC, stock_code
            """,
            (base.START_DATE, base.END_DATE),
        ).fetchall()
    finally:
        conn.close()
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        item = dict(row)
        if float(item["rank_5d"]) < float(cfg["gate_5d"]):
            continue
        by_date.setdefault(str(item["trade_date"]), []).append(item)
    signals = []
    for signal_date, items in by_date.items():
        if cfg["order_mode"] == "mix":
            items = sorted(items, key=lambda row: (0.85 * float(row["rank_10d"]) + 0.15 * float(row["rank_5d"]), row["stock_code"]), reverse=True)
        else:
            items = sorted(items, key=lambda row: (float(row["rank_10d"]), row["stock_code"]), reverse=True)
        buy_date = next_date.get(signal_date)
        if not buy_date:
            continue
        picked = 0
        for item in items:
            buy_market = market.get(buy_date, {}).get(str(item["stock_code"]))
            if base._is_st_like(buy_market) or base._is_limit_buy(buy_market):
                continue
            picked += 1
            if picked > 4:
                break
            signals.append(
                {
                    "signal_date": signal_date,
                    "buy_date": buy_date,
                    "symbol": base.to_gm_symbol(str(item["stock_code"])),
                    "stock_code": item["stock_code"],
                    "name": item.get("name"),
                    "rank": picked,
                    "pred_prob": item["pred_10d"],
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
