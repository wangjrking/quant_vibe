from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "run_pass_ratio_open_quality_boost_20260715.py"

spec = importlib.util.spec_from_file_location("open_quality_boost", BASE_SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

mod.CASES = [
    {"case": "oqb_m3_0_x105", "gap_min": -3.0, "gap_max": 0.0, "scale": 1.05},
    {"case": "oqb_m3_0_x110", "gap_min": -3.0, "gap_max": 0.0, "scale": 1.10},
    {"case": "oqb_m3_0_x115", "gap_min": -3.0, "gap_max": 0.0, "scale": 1.15},
    {"case": "oqb_m5_p05_x105", "gap_min": -5.0, "gap_max": 0.5, "scale": 1.05},
    {"case": "oqb_m5_p05_x110", "gap_min": -5.0, "gap_max": 0.5, "scale": 1.10},
    {"case": "oqb_m5_p05_x115", "gap_min": -5.0, "gap_max": 0.5, "scale": 1.15},
]
mod.SIGNAL_DIR = mod.REPORT_DIR / "signals" / "open_quality_mild"
mod.LOG_DIR = mod.REPORT_DIR / "logs" / "open_quality_mild_20260715"
mod.OUT_CSV = mod.REPORT_DIR / "pass_ratio_open_quality_mild_juejin_20260715.csv"
mod.OUT_JSON = mod.REPORT_DIR / "pass_ratio_open_quality_mild_juejin_20260715.json"
mod.OUT_MD = mod.REPORT_DIR / "pass_ratio_open_quality_mild_review_20260715.md"


if __name__ == "__main__":
    mod.main()
