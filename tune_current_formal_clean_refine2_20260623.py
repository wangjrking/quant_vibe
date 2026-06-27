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
    / "current_formal_clean_refine2"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_clean_refine2.db"
base.SCORE_DB = base.REPORT_DIR / "scores_clean_refine2.db"

base.BASE_TARGETS[4] = {1: 0.27, 2: 0.25, 3: 0.22, 4: 0.18}
base.BASE_TARGETS[6] = {1: 0.21, 2: 0.20, 3: 0.18, 4: 0.15, 5: 0.13, 6: 0.11}


def _cfg(
    name: str,
    *,
    top_k: int,
    scale: float,
    cap: float,
    holding_days: int = 5,
    score_exit: bool = False,
    exit_ratio: str = "0.95",
    min_exit_hold: int = 2,
) -> dict:
    cfg = {
        "name": name,
        "w10": 0.80,
        "w5": 0.20,
        "w3": 0.00,
        "top_k": top_k,
        "scale": scale,
        "cap": cap,
        "holding_days": holding_days,
    }
    env = {}
    if score_exit:
        env["GM_OPEN_DAILY_SCORE_EXIT"] = "1"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(exit_ratio)
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(min_exit_hold)
    else:
        env["GM_OPEN_DAILY_SCORE_EXIT"] = "0"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = "none"
    if env:
        cfg["env"] = env
    return cfg


base.VARIANTS = [
    _cfg("top5_s100_c28_h5", top_k=5, scale=1.00, cap=0.28),
    _cfg("top5_s102_c28_h5", top_k=5, scale=1.02, cap=0.28),
    _cfg("top5_s104_c28_h5", top_k=5, scale=1.04, cap=0.28),
    _cfg("top5_s106_c30_h5", top_k=5, scale=1.06, cap=0.30),
    _cfg("top5_s108_c30_h5", top_k=5, scale=1.08, cap=0.30),
    _cfg("top5_s110_c30_h5", top_k=5, scale=1.10, cap=0.30),
    _cfg("top5_s108_c30_h4", top_k=5, scale=1.08, cap=0.30, holding_days=4),
    _cfg("top5_s108_c30_h6", top_k=5, scale=1.08, cap=0.30, holding_days=6),
    _cfg("top5_s108_c30_h5_exit095_mh2", top_k=5, scale=1.08, cap=0.30, score_exit=True, exit_ratio="0.95", min_exit_hold=2),
    _cfg("top5_s108_c30_h5_exit098_mh2", top_k=5, scale=1.08, cap=0.30, score_exit=True, exit_ratio="0.98", min_exit_hold=2),
    _cfg("top5_s108_c30_h5_exit095_mh3", top_k=5, scale=1.08, cap=0.30, score_exit=True, exit_ratio="0.95", min_exit_hold=3),
    _cfg("top4_s100_c32_h5", top_k=4, scale=1.00, cap=0.32),
    _cfg("top4_s104_c32_h5", top_k=4, scale=1.04, cap=0.32),
    _cfg("top4_s108_c32_h5", top_k=4, scale=1.08, cap=0.32),
    _cfg("top4_s108_c32_h4", top_k=4, scale=1.08, cap=0.32, holding_days=4),
    _cfg("top6_s100_c26_h5", top_k=6, scale=1.00, cap=0.26),
    _cfg("top6_s104_c26_h5", top_k=6, scale=1.04, cap=0.26),
    _cfg("top6_s108_c26_h5", top_k=6, scale=1.08, cap=0.26),
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
