from __future__ import annotations

import math
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
    / "current_formal_top1keep_exit_transform_fixnorm"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_top1keep_exit_transform_fixnorm.db"
base.SCORE_DB = base.REPORT_DIR / "scores_top1keep_exit_transform_fixnorm.db"

_RANK10D_UPPER: float | None = None


def _cfg(name: str, *, transform: str, param: float | None = None) -> dict:
    return {
        "name": name,
        "w10": 0.80,
        "w5": 0.20,
        "w3": 0.00,
        "top_k": 4,
        "target_pct_each": 0.3305,
        "cap": 0.50,
        "holding_days": 5,
        "top1_rule": {
            "score_exit_entry_ratio": 0.91,
            "holding_days": 7,
        },
        "exit_transform": transform,
        "exit_param": param,
        "env": {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.935",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
        },
    }


base.VARIANTS = [
    _cfg("base_rank10d", transform="identity"),
    _cfg("pow_070", transform="pow", param=0.70),
    _cfg("pow_085", transform="pow", param=0.85),
    _cfg("pow_115", transform="pow", param=1.15),
    _cfg("pow_130", transform="pow", param=1.30),
    _cfg("tailpow_050", transform="tailpow", param=0.50),
    _cfg("tailpow_070", transform="tailpow", param=0.70),
    _cfg("tailpow_130", transform="tailpow", param=1.30),
    _cfg("tailpow_160", transform="tailpow", param=1.60),
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


def _rank10d_upper() -> float:
    global _RANK10D_UPPER
    if _RANK10D_UPPER is not None:
        return _RANK10D_UPPER
    conn = sqlite3.connect(base.FUSION_DB)
    try:
        upper = conn.execute(
            "SELECT MAX(rank_10d) FROM fusion_rank_base WHERE trade_date >= ? AND trade_date <= ?",
            (base.START_DATE, base.END_DATE),
        ).fetchone()[0]
    finally:
        conn.close()
    _RANK10D_UPPER = max(float(upper or 0.0), 1.0)
    return _RANK10D_UPPER


def _transform(value: float, kind: str, param: float | None, upper_bound: float) -> float:
    raw = float(value)
    if kind == "identity":
        return raw
    x = max(0.0, min(1.0, raw / upper_bound))
    if kind == "pow":
        gamma = float(param)
        return upper_bound * (x ** gamma)
    if kind == "tailpow":
        gamma = float(param)
        return upper_bound * (1.0 - ((1.0 - x) ** gamma))
    raise ValueError(f"unsupported transform: {kind}")


def _build_score_table(cfg: dict) -> str:
    table = base._score_table(cfg)
    base.SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    upper_bound = _rank10d_upper()
    conn = sqlite3.connect(base.SCORE_DB)
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.execute(f'CREATE TABLE "{table}" (trade_date TEXT NOT NULL, stock_code TEXT NOT NULL, pred_prob REAL)')
        src = sqlite3.connect(base.FUSION_DB)
        try:
            rows = src.execute(
                "SELECT trade_date, stock_code, rank_10d FROM fusion_rank_base WHERE trade_date >= ? AND trade_date <= ?",
                (base.START_DATE, base.END_DATE),
            ).fetchall()
        finally:
            src.close()
        payload = [
            (
                str(trade_date),
                str(stock_code),
                _transform(float(rank_10d), str(cfg["exit_transform"]), cfg.get("exit_param"), upper_bound),
            )
            for trade_date, stock_code, rank_10d in rows
            if rank_10d is not None
        ]
        conn.executemany(f'INSERT INTO "{table}" (trade_date, stock_code, pred_prob) VALUES (?, ?, ?)', payload)
        conn.execute(f'CREATE INDEX idx_{table}_trade_stock ON "{table}"(trade_date, stock_code)')
        conn.execute(f'CREATE INDEX idx_{table}_trade_pred ON "{table}"(trade_date, pred_prob DESC)')
        conn.commit()
    finally:
        conn.close()
    return table


def _build_signals(cfg: dict) -> list[dict]:
    market = base._market_rows()
    next_date = base._date_map()
    rank_expr = base._expr(cfg)
    upper_bound = _rank10d_upper()
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
        pred_prob = _transform(float(item["rank_10d"]), str(cfg["exit_transform"]), cfg.get("exit_param"), upper_bound)
        signal = {
            "signal_date": signal_date,
            "buy_date": buy_date,
            "symbol": base.to_gm_symbol(str(item["stock_code"])),
            "stock_code": item["stock_code"],
            "name": item.get("name"),
            "rank": rank,
            "pred_prob": pred_prob,
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
