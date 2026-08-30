from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "quant" / "main" / "run_current_formal_l4_independent_replace_grid_20260719.py"
SPEC = importlib.util.spec_from_file_location("independent_replace_base", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_cohort_independent_sell_20260719"
MODULE.REPORT_DIR = REPORT_DIR
MODULE.LOG_DIR = REPORT_DIR / "logs"
MODULE.OUT_CSV = REPORT_DIR / "juejin_results.csv"
MODULE.OUT_JSON = REPORT_DIR / "result.json"
MODULE.SIGNAL_FILE = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_buy_open_gap_20260719"
    / "signals"
    / "b10_m95_daily3_hold10_cohort_gap5m_to_1p.csv"
)
MODULE.SCORE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_smooth_frequency_20260719"
    / "score_assets"
    / "b10_100.duckdb"
)
MODULE.MAX_POSITIONS = 30

CASES = [
    {"min_hold": min_hold, "max_hold": max_hold, "ratio": ratio, "edge": edge}
    for min_hold in [3, 5]
    for max_hold in [10, 15]
    for ratio, edge in [(1.02, 0.005), (1.05, 0.010)]
]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODULE.LOG_DIR.mkdir(exist_ok=True)
    rows = []
    for case in CASES:
        result = MODULE.run_case(case)
        rows.append(result)
        frame = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last")
        frame.to_csv(MODULE.OUT_CSV, index=False, encoding="utf-8-sig")
        print(json.dumps({key: result.get(key) for key in ["name", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    MODULE.OUT_JSON.write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
