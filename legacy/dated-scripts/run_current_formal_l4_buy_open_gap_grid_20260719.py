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

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_buy_open_gap_20260719"
MODULE.REPORT_DIR = REPORT_DIR
MODULE.SIGNAL_DIR = REPORT_DIR / "signals"
MODULE.LOG_DIR = REPORT_DIR / "logs"

BASES = [
    {"blend": "b10", "topn": 1, "hold": 10, "model_min": 0.97},
    {"blend": "b10", "topn": 3, "hold": 10, "model_min": 0.95},
]
GAPS = [(-5.0, 1.0), (-4.0, 1.0), (-3.0, 1.0), (-3.0, 2.0), (-2.0, 1.0)]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODULE.SIGNAL_DIR.mkdir(exist_ok=True)
    MODULE.LOG_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(MODULE.POOL_DB), read_only=True)
    rows = []
    try:
        for case in BASES:
            base_name, base_signal, max_positions, target_pct = MODULE.build_signal(con, case)
            base_frame = pd.read_csv(base_signal, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str})
            for gap_min, gap_max in GAPS:
                frame = base_frame[
                    (base_frame["buy_open_gap_raw_pct"] >= gap_min)
                    & (base_frame["buy_open_gap_raw_pct"] <= gap_max)
                ].copy()
                name = f"{base_name}_gap{abs(int(gap_min))}m_to_{int(gap_max)}p"
                signal_file = MODULE.SIGNAL_DIR / f"{name}.csv"
                frame["strategy_variant"] = name
                frame["filter_name"] = "current_formal_l4_buy_open_raw_gap_gate"
                frame.to_csv(signal_file, index=False, encoding="utf-8-sig")
                result = MODULE.run_case(case, name, signal_file, max_positions, target_pct)
                result.update(
                    {
                        "gap_min": gap_min,
                        "gap_max": gap_max,
                        "signal_rows": int(len(frame)),
                        "signal_days": int(frame["signal_date"].nunique()),
                    }
                )
                rows.append(result)
                table = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last")
                table.to_csv(REPORT_DIR / "juejin_results.csv", index=False, encoding="utf-8-sig")
                print(json.dumps({key: result.get(key) for key in ["name", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    finally:
        con.close()
    (REPORT_DIR / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
