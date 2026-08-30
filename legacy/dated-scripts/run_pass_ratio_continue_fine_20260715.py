from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "run_pass_ratio_sell_rule_grid_20260715.py"

spec = importlib.util.spec_from_file_location("sell_grid", BASE_SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

mod.CASES = [
    {"case": "sell_h1_mh3_e098_c1025", "h": 1, "mh": 3, "exit": 0.98, "cont": 1.025, "minh": 1},
    {"case": "sell_h1_mh3_e098_c103", "h": 1, "mh": 3, "exit": 0.98, "cont": 1.03, "minh": 1},
    {"case": "sell_h1_mh3_e098_c1035", "h": 1, "mh": 3, "exit": 0.98, "cont": 1.035, "minh": 1},
    {"case": "sell_h1_mh3_e098_c104", "h": 1, "mh": 3, "exit": 0.98, "cont": 1.04, "minh": 1},
    {"case": "sell_h1_mh3_e098_c1045", "h": 1, "mh": 3, "exit": 0.98, "cont": 1.045, "minh": 1},
    {"case": "sell_h1_mh3_e098_c105", "h": 1, "mh": 3, "exit": 0.98, "cont": 1.05, "minh": 1},
]
mod.SIGNAL_DIR = mod.REPORT_DIR / "signals" / "continue_fine"
mod.LOG_DIR = mod.REPORT_DIR / "logs" / "continue_fine_20260715"
mod.OUT_CSV = mod.REPORT_DIR / "pass_ratio_continue_fine_juejin_20260715.csv"
mod.OUT_JSON = mod.REPORT_DIR / "pass_ratio_continue_fine_juejin_20260715.json"
mod.OUT_MD = mod.REPORT_DIR / "pass_ratio_continue_fine_review_20260715.md"


if __name__ == "__main__":
    mod.main()
