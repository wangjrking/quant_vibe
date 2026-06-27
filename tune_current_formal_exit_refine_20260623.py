from __future__ import annotations

import copy
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
    / "current_formal_exit_refine"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_exit_refine.db"
base.SCORE_DB = base.REPORT_DIR / "scores_exit_refine.db"


def _cfg(
    name: str,
    *,
    scale: float,
    cap: float = 0.30,
    holding_days: int = 5,
    exit_ratio: str = "0.95",
    min_exit_hold: int = 2,
) -> dict:
    return {
        "name": name,
        "w10": 0.80,
        "w5": 0.20,
        "w3": 0.00,
        "top_k": 5,
        "scale": scale,
        "cap": cap,
        "holding_days": holding_days,
        "env": {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": str(exit_ratio),
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(min_exit_hold),
        },
    }


base.VARIANTS = [
    _cfg("s102_exit095_mh2", scale=1.02, exit_ratio="0.95", min_exit_hold=2),
    _cfg("s104_exit095_mh2", scale=1.04, exit_ratio="0.95", min_exit_hold=2),
    _cfg("s106_exit095_mh2", scale=1.06, exit_ratio="0.95", min_exit_hold=2),
    _cfg("s108_exit093_mh2", scale=1.08, exit_ratio="0.93", min_exit_hold=2),
    _cfg("s108_exit094_mh2", scale=1.08, exit_ratio="0.94", min_exit_hold=2),
    _cfg("s108_exit095_mh2", scale=1.08, exit_ratio="0.95", min_exit_hold=2),
    _cfg("s108_exit096_mh2", scale=1.08, exit_ratio="0.96", min_exit_hold=2),
    _cfg("s108_exit097_mh2", scale=1.08, exit_ratio="0.97", min_exit_hold=2),
    _cfg("s108_exit095_mh1", scale=1.08, exit_ratio="0.95", min_exit_hold=1),
    _cfg("s108_exit095_mh3", scale=1.08, exit_ratio="0.95", min_exit_hold=3),
    _cfg("s110_exit095_mh2", scale=1.10, exit_ratio="0.95", min_exit_hold=2),
    _cfg("s112_exit095_mh2", scale=1.12, exit_ratio="0.95", min_exit_hold=2),
    _cfg("s108_exit095_mh2_h4", scale=1.08, holding_days=4, exit_ratio="0.95", min_exit_hold=2),
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
