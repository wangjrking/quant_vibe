from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "quant" / "main" / "run_current_formal_l4_rolling_cohort_grid_20260719.py"
SPEC = importlib.util.spec_from_file_location("rolling_cohort_base", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_rolling_cohort_refine_20260719"
MODULE.REPORT_DIR = REPORT_DIR
MODULE.SIGNAL_DIR = REPORT_DIR / "signals"
MODULE.LOG_DIR = REPORT_DIR / "logs"

CASES = [
    {"blend": "b10", "topn": topn, "hold": 10, "model_min": model_min}
    for topn in [1, 2, 3]
    for model_min in [0.97, 0.98, 0.99]
] + [
    {"blend": "b10", "topn": 3, "hold": hold, "model_min": 0.98}
    for hold in [12, 15]
]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODULE.SIGNAL_DIR.mkdir(exist_ok=True)
    MODULE.LOG_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(MODULE.POOL_DB), read_only=True)
    rows = []
    try:
        for case in CASES:
            name, signal_file, max_positions, target_pct = MODULE.build_signal(con, case)
            result = MODULE.run_case(case, name, signal_file, max_positions, target_pct)
            rows.append(result)
            frame = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last")
            frame.to_csv(REPORT_DIR / "juejin_results.csv", index=False, encoding="utf-8-sig")
            print(json.dumps({key: result.get(key) for key in ["name", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    finally:
        con.close()
    (REPORT_DIR / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
