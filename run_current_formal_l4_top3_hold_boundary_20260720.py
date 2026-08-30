from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "quant" / "main" / "run_current_formal_l4_5d10d_top3_independent_sell_refine_20260720.py"
SPEC = importlib.util.spec_from_file_location("top3_refine", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top3_hold_boundary_20260720"
MODULE.OUT = OUT
MODULE.SIGNALS = OUT / "signals"
MODULE.LOGS = OUT / "logs"
MODULE.CASES = [
    (rank1, min_hold, max_hold, 1.0, 0.0)
    for rank1 in (0.86, 0.88, 0.90)
    for min_hold, max_hold in ((12, 12), (13, 13), (13, 14))
]


def main() -> None:
    for path in (MODULE.OUT, MODULE.SIGNALS, MODULE.LOGS):
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in MODULE.CASES:
        result = MODULE.run_case(*case)
        rows.append(result)
        pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False, na_position="last").to_csv(
            MODULE.OUT / "juejin_results.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({key: result.get(key) for key in ("name", "returncode", "annual_return", "sharpe", "max_drawdown")}, ensure_ascii=False), flush=True)
    (MODULE.OUT / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
