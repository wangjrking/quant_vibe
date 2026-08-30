from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PICK_SCRIPT = ROOT / "quant" / "main" / "research_top3_multi_open_quality_sizing_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_multi_scaled_lowrisk"
LOG_DIR = REPORT_DIR / "logs" / "top3_multi_scaled_lowrisk"


spec = importlib.util.spec_from_file_location("multi", PICK_SCRIPT)
multi_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(multi_mod)


def main() -> None:
    base = multi_mod.pick_mod.load_all()
    multi_mod.pick_mod.base_mod.SIGNAL_DIR = SIGNAL_DIR
    multi_mod.pick_mod.base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    base_weights = (0.60, 0.30, 0.18)
    for top_n in [2, 3]:
        for scale in [1.10, 1.20, 1.30, 1.40, 1.50, 1.60]:
            high, mid, low = [x * scale for x in base_weights]
            for daily_cap in [0.91, 0.98]:
                cases.append(
                    {
                        "case": (
                            f"ms_t{top_n}_sc{int(scale*100):03d}_cap{int(daily_cap*100):02d}"
                        ),
                        "top_n": top_n,
                        "high": high,
                        "mid": mid,
                        "low": low,
                        "shallow_scale": 0.75,
                        "pos_gap_scale": 0.50,
                        "deep_scale": 1.00,
                        "deep_gap_scale": 1.05,
                        "daily_cap": daily_cap,
                    }
                )

    manifest = [multi_mod.write_case(base, case) for case in cases]
    results = []
    for row in manifest:
        result = multi_mod.pick_mod.base_mod.run_juejin(row)
        results.append(result)
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "pnl_ratio_annual": result.get("pnl_ratio_annual"),
                    "sharp_ratio": result.get("sharp_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    csv_path = REPORT_DIR / "top3_multi_scaled_lowrisk_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_multi_scaled_lowrisk_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
