from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
BASE_SCRIPT = MAIN / "run_adaptive70w_latest_top3_frequency_grid_20260630.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("adaptive70w_latest_l4_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = _load_base()
    base.REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_20260701_quick_frequency_grid_20260702"
    base.STRATEGY_DIR = base.REPORT_DIR / "code_snapshot_dynamic_slippage_duckdb_fetchall_20260702"
    base.SCORE_DB = base.REPORT_DIR / "scores" / "grid_scores_20260702_quick.duckdb"
    base.JUEJIN_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
    base.SLICES = [("full", "2022-06-07 09:00:00", "2026-07-01 15:30:00")]
    base.TOP_CASES = [
        {"name": "top1_pos45", "topn": 1, "max_positions": 1, "target": 0.45},
        {"name": "top1_pos50", "topn": 1, "max_positions": 1, "target": 0.50},
        {"name": "top1_pos55", "topn": 1, "max_positions": 1, "target": 0.55},
        {"name": "top1_pos60", "topn": 1, "max_positions": 1, "target": 0.60},
        {"name": "top1_pos65", "topn": 1, "max_positions": 1, "target": 0.65},
        {"name": "top1_pos70", "topn": 1, "max_positions": 1, "target": 0.70},
    ]
    base.BUY_RULES = [
        {"name": "cool2d20", "two_day_cap": 0.20, "combo_cap": None, "turnover_floor": None},
    ]
    base.SELL_RULES = [
        {
            "name": "h2m3_e097_c098_min1",
            "holding_days": 2,
            "max_holding_days": 3,
            "score_exit_ratio": 0.97,
            "score_continue_ratio": 0.98,
            "min_score_exit_days": 1,
            "day_drop_ratio": None,
        },
    ]
    base.RISK_MODES = [
        {
            "name": "open_only_ddloose",
            "intraday": "0",
            "max_daily_sells": "0",
            "dd_soft": "0.08",
            "dd_hard": "0.14",
            "dd_recover": "0.04",
            "dd_soft_scale": "0.80",
            "dd_hard_scale": "0.60",
        }
    ]

    def score_table_ready_20260701(table: str) -> bool:
        return False

    base._score_table_ready = score_table_ready_20260701
    base.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
