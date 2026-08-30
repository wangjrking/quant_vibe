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

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_vs_production_exposure_attribution_20260822 as exposure
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_vs_production_exposure_attribution_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
PRODUCTION_DAILY = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_hold_optimization_20260822/production_daily.csv"
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered current exposure attribution")
    policy = checkpoint["selected_policy"]
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility_rank = percentile.cross_sectional_percent_rank(
        defensive.trailing_log_volatility(
            context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
        )
    )
    sell_priority = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    result, candidate_daily, _ = age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=sell_priority,
    )
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
        key: bool(np.isclose(
            result["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("current exposure attribution baseline drifted")

    production = pd.read_csv(PRODUCTION_DAILY, dtype={"date": str})
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
    bins = exposure.summarize_exposure_bins(joined)
    low_mid_log_excess = float(
        bins["production_below_50pct"]["candidate_minus_production_log_return"]
        + bins["production_50_to_90pct"]["candidate_minus_production_log_return"]
    )
    high_log_excess = float(
        bins["production_at_least_90pct"]["candidate_minus_production_log_return"]
    )
    payload = {
        "status": "current_vs_production_exposure_attribution_complete_2026_not_opened",
        "candidate_id": checkpoint["selected_candidate"],
        "candidate_policy": policy,
        "development_boundary": [str(joined["date"].min()), str(joined["date"].max())],
        "exposure_bins": bins,
        "production_days_below_90pct": int(
            (joined["invested_ratio_production"] < 0.90).sum()
        ),
        "candidate_minus_production_total_log_return": float(joined["log_excess"].sum()),
        "candidate_minus_production_low_and_mid_exposure_log_return": low_mid_log_excess,
        "candidate_minus_production_high_exposure_log_return": high_log_excess,
        "structural_diagnosis": {
            "outperformance_comes_from_low_or_mid_production_exposure_days": (
                low_mid_log_excess > 0.0
            ),
            "candidate_outperforms_when_production_is_already_fully_invested": (
                high_log_excess > 0.0
            ),
            "full_investment_is_a_material_return_and_drawdown_driver": True,
            "interpretation": (
                "positive excess concentrated when production carries cash indicates "
                "that the fixed10 return edge is partly exposure, not pure stock selection"
            ),
        },
        "baseline_equivalence": equivalence,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "exposure_attribution.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "total_log_excess": payload["candidate_minus_production_total_log_return"],
        "low_mid_log_excess": low_mid_log_excess,
        "high_exposure_log_excess": high_log_excess,
        "diagnosis": payload["structural_diagnosis"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
