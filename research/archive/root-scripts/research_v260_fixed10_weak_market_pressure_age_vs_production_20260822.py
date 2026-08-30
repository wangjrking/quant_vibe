from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_extra_age_regime_decomposition_20260822 as decomposition
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_pressure_vs_production_exposure_attribution_20260822 as old


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_market_pressure_age_vs_production_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak_market_pressure_age_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered exposure attribution")
    policy = checkpoint["selected_policy"]
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    schedule = decomposition.age_schedule(strong, 4, 8)
    candidate_daily, _ = age_guard.run_policy_at_cost(
        context, policy, schedule, round1.BASELINE_COST
    )
    production = pd.read_csv(old.PRODUCTION_DAILY, dtype={"date": str})
    candidate = candidate_daily.copy()
    candidate["date"] = candidate["date"].astype(str)
    joined = production[["date", "return", "invested_ratio"]].merge(
        candidate[["date", "return", "invested_ratio"]],
        on="date",
        suffixes=("_production", "_candidate"),
        validate="one_to_one",
    )
    joined["log_excess"] = np.log1p(joined["return_candidate"]) - np.log1p(
        joined["return_production"]
    )
    bins = old.summarize_exposure_bins(joined)
    result = {
        "status": "weak_market_pressure_age_vs_production_complete_2026_not_opened",
        "development_boundary": [str(joined["date"].min()), str(joined["date"].max())],
        "candidate_policy": policy,
        "exposure_bins": bins,
        "production_days_below_90pct": int((joined["invested_ratio_production"] < 0.90).sum()),
        "candidate_minus_production_total_log_return": float(joined["log_excess"].sum()),
        "candidate_minus_production_on_low_exposure_days": float(
            joined.loc[joined["invested_ratio_production"] < 0.50, "log_excess"].sum()
        ),
        "candidate_minus_production_on_other_days": float(
            joined.loc[joined["invested_ratio_production"] >= 0.50, "log_excess"].sum()
        ),
        "structural_diagnosis": {
            "candidate_loses_low_exposure_regime": bool(
                bins["production_below_50pct"]["candidate_minus_production_log_return"] < 0.0
            ),
            "candidate_wins_when_production_not_low_exposure": bool(
                joined.loc[joined["invested_ratio_production"] >= 0.50, "log_excess"].sum() > 0.0
            ),
            "full_investment_constraint_is_structural": True,
        },
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "exposure_attribution.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
