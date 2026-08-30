from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top1_three_tier_open_quality_refine"
LOG_DIR = REPORT_DIR / "logs" / "top1_three_tier_open_quality_refine"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


def main() -> None:
    base = base_mod.load_top1()
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR

    cases = []
    profiles = ["mid_atr12", "mid_deep_only", "mid_gap_amt"]
    weight_sets = [
        (0.79, 0.36, 0.16),
        (0.78, 0.36, 0.18),
        (0.77, 0.36, 0.20),
        (0.76, 0.36, 0.22),
        (0.75, 0.36, 0.24),
        (0.74, 0.36, 0.26),
        (0.78, 0.34, 0.20),
        (0.76, 0.34, 0.22),
        (0.74, 0.34, 0.24),
        (0.72, 0.34, 0.26),
        (0.70, 0.34, 0.28),
        (0.78, 0.32, 0.22),
        (0.76, 0.32, 0.24),
        (0.74, 0.32, 0.26),
        (0.72, 0.32, 0.28),
        (0.70, 0.32, 0.30),
        (0.76, 0.30, 0.26),
        (0.74, 0.30, 0.28),
        (0.72, 0.30, 0.30),
    ]
    for profile in profiles:
        for high, mid, low in weight_sets:
            cases.append(
                {
                    "case": f"tier2_{profile}_h{int(high*100):02d}_m{int(mid*100):02d}_l{int(low*100):02d}",
                    "profile": profile,
                    "high": high,
                    "mid": mid,
                    "low": low,
                }
            )

    manifest = [base_mod.write_signal(base, case) for case in cases]
    results = [base_mod.run_juejin(row) for row in manifest]
    csv_path = REPORT_DIR / "top1_three_tier_open_quality_refine_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top1_three_tier_open_quality_refine_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
