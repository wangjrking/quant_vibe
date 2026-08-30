from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PICK_SCRIPT = ROOT / "quant" / "main" / "research_top3_open_quality_pick1_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_open_quality_pick1_lowrisk"
LOG_DIR = REPORT_DIR / "logs" / "top3_open_quality_pick1_lowrisk"


spec = importlib.util.spec_from_file_location("pick1", PICK_SCRIPT)
pick_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(pick_mod)


def main() -> None:
    base = pick_mod.load_all()
    pick_mod.base_mod.SIGNAL_DIR = SIGNAL_DIR
    pick_mod.base_mod.LOG_DIR = LOG_DIR
    cases = []
    weights = [
        (0.72, 0.36, 0.24),
        (0.70, 0.36, 0.26),
        (0.68, 0.36, 0.28),
        (0.66, 0.36, 0.30),
        (0.64, 0.36, 0.32),
        (0.72, 0.34, 0.26),
        (0.70, 0.34, 0.28),
        (0.68, 0.34, 0.30),
        (0.66, 0.34, 0.32),
        (0.64, 0.34, 0.34),
        (0.70, 0.32, 0.30),
        (0.68, 0.32, 0.32),
        (0.66, 0.32, 0.34),
    ]
    for profile in ["mid_atr12", "mid_deep_only"]:
        for high, mid, low in weights:
            cases.append(
                {
                    "case": f"pick1low_{profile}_h{int(high*100):02d}_m{int(mid*100):02d}_l{int(low*100):02d}",
                    "profile": profile,
                    "sort_by": "quality_pred10",
                    "high": high,
                    "mid": mid,
                    "low": low,
                }
            )
    manifest = [pick_mod.write_pick_signal(base, case) for case in cases]
    results = [pick_mod.base_mod.run_juejin(row) for row in manifest]
    csv_path = REPORT_DIR / "top3_open_quality_pick1_lowrisk_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_open_quality_pick1_lowrisk_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
