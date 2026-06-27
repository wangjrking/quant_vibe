from __future__ import annotations

import sqlite3
from typing import Dict

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
    / "current_formal_flat_profile_refine"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_flat_profile_refine.db"
base.SCORE_DB = base.REPORT_DIR / "scores_flat_profile_refine.db"


def _cfg(
    name: str,
    *,
    targets: Dict[int, float],
    scale: float,
    cap: float,
    exit_ratio: str = "0.94",
    exit_rank: str | None = "0.94",
) -> dict:
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
        "targets": targets,
        "scale": scale,
        "cap": cap,
        "holding_days": 5,
        "env": env,
    }


base.VARIANTS = [
    _cfg("flat025_s120_cap34", targets={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, scale=1.20, cap=0.34),
    _cfg("flat025_s123_cap35", targets={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, scale=1.23, cap=0.35),
    _cfg("flat025_s125_cap36", targets={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, scale=1.25, cap=0.36),
    _cfg("flat025_s127_cap37", targets={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, scale=1.27, cap=0.37),
    _cfg("flat025_s130_cap38", targets={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, scale=1.30, cap=0.38),
    _cfg("flat025_s133_cap39", targets={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, scale=1.33, cap=0.39),
    _cfg("flat024_s130_cap37", targets={1: 0.24, 2: 0.24, 3: 0.24, 4: 0.24}, scale=1.30, cap=0.37),
    _cfg("flat026_s120_cap37", targets={1: 0.26, 2: 0.26, 3: 0.26, 4: 0.26}, scale=1.20, cap=0.37),
    _cfg("flat026_s123_cap38", targets={1: 0.26, 2: 0.26, 3: 0.26, 4: 0.26}, scale=1.23, cap=0.38),
    _cfg("flat026_s125_cap39", targets={1: 0.26, 2: 0.26, 3: 0.26, 4: 0.26}, scale=1.25, cap=0.39),
    _cfg("flat025_s125_cap36_exit093", targets={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, scale=1.25, cap=0.36, exit_ratio="0.93"),
    _cfg("flat025_s125_cap36_norank", targets={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, scale=1.25, cap=0.36, exit_rank=None),
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


def _build_signals(cfg: dict) -> list[dict]:
    market = base._market_rows()
    next_date = base._date_map()
    expr = base._expr(cfg)
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
                    {expr} AS score,
                    ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY {expr} DESC, stock_code) AS rn
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
    targets = {int(k): float(v) for k, v in dict(cfg["targets"]).items()}
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
        raw_target = float(targets.get(rank, 0.0))
        target = min(raw_target * float(cfg["scale"]), float(cfg["cap"]))
        signals.append(
            {
                "signal_date": signal_date,
                "buy_date": buy_date,
                "symbol": base.to_gm_symbol(str(item["stock_code"])),
                "stock_code": item["stock_code"],
                "name": item.get("name"),
                "rank": rank,
                "pred_prob": item["score"],
                "pred_5d": item.get("pred_5d"),
                "pred_10d": item.get("pred_10d"),
                "rank_5d": item.get("rank_5d"),
                "rank_10d": item.get("rank_10d"),
                "target_pct": f"{target:.5f}",
                "holding_days": int(cfg["holding_days"]),
                "max_holding_days": int(cfg["holding_days"]),
            }
        )
    return signals


base._build_fusion_db = _build_fusion_db
base._build_signals = _build_signals


if __name__ == "__main__":
    raise SystemExit(base.main())
