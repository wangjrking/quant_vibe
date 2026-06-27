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
    / "current_formal_top4_micro_refine"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_top4_micro_refine.db"
base.SCORE_DB = base.REPORT_DIR / "scores_top4_micro_refine.db"

base.BASE_TARGETS[4] = {1: 0.27, 2: 0.25, 3: 0.22, 4: 0.18}


def _cfg(
    name: str,
    *,
    scale: float,
    cap: float,
    exit_ratio: str = "0.94",
    exit_rank: str | None = "0.94",
    day_drop_ratio: str | None = None,
) -> dict:
    env = {
        "GM_OPEN_DAILY_SCORE_EXIT": "1",
        "GM_SCORE_EXIT_ENTRY_RATIO": str(exit_ratio),
        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
    }
    if exit_rank is not None:
        env["GM_SCORE_EXIT_RANK"] = str(exit_rank)
    if day_drop_ratio is not None:
        env["GM_SCORE_STOP_LOSS_DAY_DROP_RATIO"] = str(day_drop_ratio)
    return {
        "name": name,
        "w10": 0.80,
        "w5": 0.20,
        "w3": 0.00,
        "top_k": 4,
        "scale": scale,
        "cap": cap,
        "holding_days": 5,
        "env": env,
    }


base.VARIANTS = [
    _cfg("top4_s1285_exit094_rank094", scale=1.285, cap=0.412),
    _cfg("top4_s1290_exit094_rank094", scale=1.290, cap=0.415),
    _cfg("top4_s1295_exit094_rank094", scale=1.295, cap=0.417),
    _cfg("top4_s1300_exit094_rank094", scale=1.300, cap=0.420),
    _cfg("top4_s1290_exit093_rank094", scale=1.290, cap=0.415, exit_ratio="0.93"),
    _cfg("top4_s1290_exit094_norank", scale=1.290, cap=0.415, exit_rank=None),
    _cfg("top4_s1290_exit094_rank096", scale=1.290, cap=0.415, exit_rank="0.96"),
    _cfg("top4_s1290_exit094_rank094_dd995", scale=1.290, cap=0.415, day_drop_ratio="0.995"),
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
