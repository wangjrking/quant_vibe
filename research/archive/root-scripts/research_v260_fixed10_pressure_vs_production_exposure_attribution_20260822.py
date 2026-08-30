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
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_vs_production_exposure_attribution_20260822"
)
PRESSURE_CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
PRODUCTION_DAILY = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_hold_optimization_20260822/production_daily.csv"
)
EXPOSURE_BINS = (
    ("production_below_50pct", -np.inf, 0.50),
    ("production_50_to_90pct", 0.50, 0.90),
    ("production_at_least_90pct", 0.90, np.inf),
)


def compound_return(values) -> float:
    return float(np.prod(1.0 + np.asarray(values, dtype=np.float64)) - 1.0)


def summarize_exposure_bins(joined: pd.DataFrame) -> dict:
    result = {}
    for label, lower, upper in EXPOSURE_BINS:
        selected = joined[
            (joined["invested_ratio_production"] >= lower)
            & (joined["invested_ratio_production"] < upper)
        ]
        result[label] = {
            "days": int(len(selected)),
            "production_average_invested_ratio": float(
                selected["invested_ratio_production"].mean()
            ),
            "candidate_compound_return": compound_return(
                selected["return_candidate"]
            ),
            "production_compound_return": compound_return(
                selected["return_production"]
            ),
            "candidate_minus_production_log_return": float(
                selected["log_excess"].sum()
            ),
        }
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(PRESSURE_CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered exposure attribution")
    policy = checkpoint["selected_policy"]
    context = harness.load_context(policy)
    _, candidate_daily, _ = harness.run_policy(context, policy)
    production_daily = pd.read_csv(PRODUCTION_DAILY, dtype={"date": str})
    candidate = candidate_daily.copy()
    candidate["date"] = candidate["date"].astype(str)
    joined = production_daily[["date", "return", "invested_ratio"]].merge(
        candidate[["date", "return", "invested_ratio"]],
        on="date",
        suffixes=("_production", "_candidate"),
        validate="one_to_one",
    )
    joined["log_excess"] = np.log1p(joined["return_candidate"]) - np.log1p(
        joined["return_production"]
    )
    bins = summarize_exposure_bins(joined)
    result = {
        "status": "pressure_vs_production_exposure_attribution_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [str(joined["date"].min()), str(joined["date"].max())],
        "candidate_policy": policy,
        "exposure_bins": bins,
        "production_days_below_90pct": int(
            (joined["invested_ratio_production"] < 0.90).sum()
        ),
        "candidate_minus_production_total_log_return": float(
            joined["log_excess"].sum()
        ),
        "candidate_minus_production_first_20_days_log_return": float(
            joined.head(20)["log_excess"].sum()
        ),
        "candidate_minus_production_after_first_20_days_log_return": float(
            joined.iloc[20:]["log_excess"].sum()
        ),
        "structural_diagnosis": {
            "candidate_wins_mid_exposure_regime": bool(
                bins["production_50_to_90pct"][
                    "candidate_minus_production_log_return"
                ]
                > 0.0
            ),
            "candidate_loses_low_exposure_regime": bool(
                bins["production_below_50pct"][
                    "candidate_minus_production_log_return"
                ]
                < 0.0
            ),
            "full_investment_constraint_prevents_copying_production_derisking": True,
        },
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "exposure_attribution.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
