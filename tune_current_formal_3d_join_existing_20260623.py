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
    / "current_formal_3d_join_existing"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_3d_join_existing.db"
base.SCORE_DB = base.REPORT_DIR / "scores_3d_join_existing.db"


base.VARIANTS = [
    {"name": "raw_w10_80_w5_20_w3_00_top5_s108_h5", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 5, "scale": 1.08, "cap": 0.30, "holding_days": 5},
    {"name": "raw_w10_80_w5_20_w3_00_top5_s110_h5", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 5, "scale": 1.10, "cap": 0.30, "holding_days": 5},
    {"name": "raw_w10_80_w5_20_w3_00_top5_s112_h5", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "raw_w10_80_w5_20_w3_00_top5_s114_h5", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 5, "scale": 1.14, "cap": 0.30, "holding_days": 5},
    {"name": "raw_w10_80_w5_20_w3_00_top5_s116_h5", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 5, "scale": 1.16, "cap": 0.30, "holding_days": 5},
    {"name": "raw_w10_80_w5_20_w3_00_top5_s118_h5", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 5, "scale": 1.18, "cap": 0.32, "holding_days": 5},
    {"name": "raw_w10_80_w5_20_w3_00_top5_s120_h5", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 5, "scale": 1.20, "cap": 0.32, "holding_days": 5},
    {"name": "raw_w10_78_w5_20_w3_02_top5_s112_h5", "w10": 0.78, "w5": 0.20, "w3": 0.02, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "raw_w10_75_w5_20_w3_05_top5_s112_h5", "w10": 0.75, "w5": 0.20, "w3": 0.05, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "raw_w10_70_w5_20_w3_10_top5_s112_h5", "w10": 0.70, "w5": 0.20, "w3": 0.10, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "raw_w10_65_w5_20_w3_15_top5_s112_h5", "w10": 0.65, "w5": 0.20, "w3": 0.15, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "raw_w10_70_w5_10_w3_20_top5_s112_h5", "w10": 0.70, "w5": 0.10, "w3": 0.20, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "raw_w10_80_w5_20_w3_00_top2_h5", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 2, "scale": 1.00, "cap": 0.98, "holding_days": 5},
    {"name": "raw_w10_80_w5_20_w3_00_top2_h4", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 2, "scale": 1.00, "cap": 0.98, "holding_days": 4},
    {"name": "raw_w10_80_w5_20_w3_00_top2_h6", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 2, "scale": 1.00, "cap": 0.98, "holding_days": 6},
    {"name": "raw_w10_75_w5_20_w3_05_top2_h5", "w10": 0.75, "w5": 0.20, "w3": 0.05, "top_k": 2, "scale": 1.00, "cap": 0.98, "holding_days": 5},
    {"name": "raw_w10_70_w5_20_w3_10_top2_h5", "w10": 0.70, "w5": 0.20, "w3": 0.10, "top_k": 2, "scale": 1.00, "cap": 0.98, "holding_days": 5},
]


def _build_fusion_db() -> None:
    base.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if base.FUSION_DB.exists():
        return
    conn = sqlite3.connect(base.FUSION_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS old", (str(OLD_FUSION_DB),))
        conn.execute("ATTACH DATABASE ? AS model", (str(base.MODEL_DB),))
        conn.executescript(
            f"""
            CREATE TABLE fusion_rank_base AS
            SELECT
                old.trade_date,
                old.stock_code,
                t3.pred_prob AS pred_3d,
                old.pred_5d,
                old.pred_10d,
                t3.pred_prob AS rank_3d,
                old.rank_5d,
                old.rank_10d,
                old.name,
                old.pre_close,
                old.open,
                old.close,
                old.amount,
                old.turnover_rate,
                old.total_mv,
                old.atr_qfq,
                old.limit_times,
                old.st_type,
                old.st_type_name
            FROM old.fusion_rank_base old
            JOIN model."{base.TABLE_3D}" t3
              ON old.trade_date = t3.trade_date AND old.stock_code = t3.stock_code;
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
