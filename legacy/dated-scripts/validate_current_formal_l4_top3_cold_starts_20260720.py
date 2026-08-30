from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "quant" / "main" / "validate_current_formal_l4_5d10d_top3_independent_sell_20260720.py"
SPEC = importlib.util.spec_from_file_location("top3_validation", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_top3_cold_start_validation_20260720"
MODULE.OUT = OUT
MODULE.LOGS = OUT / "logs"
END = "2026-07-17 15:30:00"
WINDOWS = [
    ("anchor_202407_a", "2024-06-27 09:00:00", END),
    ("anchor_202407_b", "2024-07-01 09:00:00", END),
    ("anchor_202407_c", "2024-07-03 09:00:00", END),
    ("anchor_202507_a", "2025-06-27 09:00:00", END),
    ("anchor_202507_b", "2025-07-01 09:00:00", END),
    ("anchor_202507_c", "2025-07-03 09:00:00", END),
    ("anchor_202601_a", "2025-12-30 09:00:00", END),
    ("anchor_202601_b", "2026-01-05 09:00:00", END),
    ("anchor_202601_c", "2026-01-07 09:00:00", END),
]


def main() -> None:
    MODULE.LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for window in WINDOWS:
        result = MODULE.run_window(*window)
        rows.append(result)
        pd.DataFrame(rows).to_csv(OUT / "juejin_results.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
