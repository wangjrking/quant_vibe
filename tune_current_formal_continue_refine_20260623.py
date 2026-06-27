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
    / "current_formal_continue_refine"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_continue_refine.db"
base.SCORE_DB = base.REPORT_DIR / "scores_continue_refine.db"


def _cfg(name: str, *, scale: float, exit_ratio: str, continue_ratio: str, max_holding_days: int) -> dict:
    return {
        "name": name,
        "w10": 0.80,
        "w5": 0.20,
        "w3": 0.00,
        "top_k": 5,
        "scale": scale,
        "cap": 0.30,
        "holding_days": 5,
        "env": {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": str(exit_ratio),
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
            "GM_MAX_HOLDING_DAYS": str(max_holding_days),
            "GM_SCORE_CONTINUE_ENTRY_RATIO": str(continue_ratio),
        },
    }


base.VARIANTS = [
    _cfg("s108_exit093_continue098_max7", scale=1.08, exit_ratio="0.93", continue_ratio="0.98", max_holding_days=7),
    _cfg("s108_exit093_continue100_max7", scale=1.08, exit_ratio="0.93", continue_ratio="1.00", max_holding_days=7),
    _cfg("s108_exit093_continue102_max7", scale=1.08, exit_ratio="0.93", continue_ratio="1.02", max_holding_days=7),
    _cfg("s108_exit093_continue105_max7", scale=1.08, exit_ratio="0.93", continue_ratio="1.05", max_holding_days=7),
    _cfg("s108_exit093_continue098_max8", scale=1.08, exit_ratio="0.93", continue_ratio="0.98", max_holding_days=8),
    _cfg("s108_exit093_continue100_max8", scale=1.08, exit_ratio="0.93", continue_ratio="1.00", max_holding_days=8),
    _cfg("s108_exit093_continue102_max8", scale=1.08, exit_ratio="0.93", continue_ratio="1.02", max_holding_days=8),
    _cfg("s108_exit093_continue105_max8", scale=1.08, exit_ratio="0.93", continue_ratio="1.05", max_holding_days=8),
    _cfg("s108_exit093_continue098_max10", scale=1.08, exit_ratio="0.93", continue_ratio="0.98", max_holding_days=10),
    _cfg("s108_exit093_continue100_max10", scale=1.08, exit_ratio="0.93", continue_ratio="1.00", max_holding_days=10),
    _cfg("s108_exit093_continue102_max10", scale=1.08, exit_ratio="0.93", continue_ratio="1.02", max_holding_days=10),
    _cfg("s108_exit093_continue105_max10", scale=1.08, exit_ratio="0.93", continue_ratio="1.05", max_holding_days=10),
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


base._build_fusion_db = _build_fusion_db


if __name__ == "__main__":
    raise SystemExit(base.main())
