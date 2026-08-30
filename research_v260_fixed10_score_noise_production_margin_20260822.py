from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1


REPORTS = REPO / "quant/data_file/reports"
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_score_noise_production_margin_20260822"
)
NOISE = (
    REPORTS
    / "strategy_agent_v260_fixed10_small_score_noise_robustness_20260822/"
    "small_score_noise_robustness.json"
)
CHECKPOINT = (
    REPORTS
    / "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    noise = json.loads(NOISE.read_text(encoding="utf-8"))
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if noise["validation_2026_opened"] or checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered score-noise production margin")
    production = checkpoint["production_baseline"]
    comparisons = {}
    for sigma, runs in noise["raw_runs"].items():
        comparisons[sigma] = {
            "runs": int(len(runs)),
            "fraction_cumulative_return_above_production": float(np.mean([
                run["cumulative_return"] > production["cumulative_return"]
                for run in runs
            ])),
            "fraction_cagr_above_production": float(np.mean([
                run["cagr"] > production["cagr"] for run in runs
            ])),
            "fraction_sharpe_above_production": float(np.mean([
                run["sharpe"] > production["sharpe"] for run in runs
            ])),
            "fraction_drawdown_not_worse_than_production": float(np.mean([
                run["max_drawdown"] <= production["max_drawdown"] for run in runs
            ])),
            "median_cumulative_return_margin": float(
                np.median([run["cumulative_return"] for run in runs])
                - production["cumulative_return"]
            ),
            "median_sharpe_margin": float(
                np.median([run["sharpe"] for run in runs])
                - production["sharpe"]
            ),
        }
    result = {
        "status": "score_noise_production_margin_complete_2026_not_opened",
        "production_reference": production,
        "comparisons": comparisons,
        "interpretation": (
            "The fixed10 candidate has a return-oriented in-sample edge, but it does "
            "not dominate production on Sharpe or drawdown and the return margin is "
            "sensitive to realistic top10 rank perturbations. One-shot 2026 validation "
            "must therefore report return, Sharpe and drawdown together."
        ),
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "score_noise_production_margin.json", result)
    print(json.dumps({
        "status": result["status"],
        "comparisons": comparisons,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
