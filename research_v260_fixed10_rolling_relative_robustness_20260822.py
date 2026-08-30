from __future__ import annotations

import copy
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
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rolling_relative_robustness_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
WINDOWS = (21, 63, 126, 252)


def run_current(context, policy: dict):
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        round1.BASELINE_COST,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_priority_override=priority_matrix,
    )


def aligned_returns(current, production) -> pd.DataFrame:
    left = current.loc[
        (current["date"].astype(str) >= research_base.FIRST_BUY)
        & (current["date"].astype(str) <= round1.DEVELOPMENT_END),
        ["date", "return"],
    ].rename(columns={"return": "fixed10"})
    right = production.loc[
        (production["date"].astype(str) >= research_base.FIRST_BUY)
        & (production["date"].astype(str) <= round1.DEVELOPMENT_END),
        ["date", "return"],
    ].rename(columns={"return": "production"})
    aligned = left.merge(right, on="date", how="inner", validate="one_to_one")
    if len(aligned) != len(left) or len(aligned) != len(right):
        raise RuntimeError("rolling comparison calendars do not align")
    return aligned.reset_index(drop=True)


def rolling_summary(frame: pd.DataFrame, window: int) -> dict:
    fixed = frame["fixed10"].to_numpy(dtype=np.float64)
    production = frame["production"].to_numpy(dtype=np.float64)
    rows = []
    for end in range(window - 1, len(frame)):
        start = end - window + 1
        fixed_return = float(np.prod(1.0 + fixed[start : end + 1]) - 1.0)
        production_return = float(
            np.prod(1.0 + production[start : end + 1]) - 1.0
        )
        rows.append({
            "start": str(frame.iloc[start]["date"]),
            "end": str(frame.iloc[end]["date"]),
            "fixed10_return": fixed_return,
            "production_return": production_return,
            "excess_return": fixed_return - production_return,
        })
    excess = np.array([row["excess_return"] for row in rows], dtype=np.float64)
    worst = min(rows, key=lambda row: (row["excess_return"], row["start"]))
    best = max(rows, key=lambda row: (row["excess_return"], row["start"]))
    return {
        "window_sessions": int(window),
        "window_count": int(len(rows)),
        "fixed10_outperformance_frequency": float((excess > 0).mean()),
        "median_excess_return": float(np.median(excess)),
        "mean_excess_return": float(excess.mean()),
        "p10_excess_return": float(np.quantile(excess, 0.10)),
        "p90_excess_return": float(np.quantile(excess, 0.90)),
        "worst_window": worst,
        "best_window": best,
    }


def calendar_period_summary(frame: pd.DataFrame, frequency: str) -> list[dict]:
    dates = pd.to_datetime(frame["date"], format="%Y%m%d")
    periods = dates.dt.to_period(frequency)
    rows = []
    for period in sorted(periods.unique()):
        mask = periods == period
        fixed = float(np.prod(1.0 + frame.loc[mask, "fixed10"]) - 1.0)
        production = float(np.prod(1.0 + frame.loc[mask, "production"]) - 1.0)
        rows.append({
            "period": str(period),
            "sessions": int(mask.sum()),
            "fixed10_return": fixed,
            "production_return": production,
            "excess_return": fixed - production,
        })
    return rows


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered rolling robustness")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    current_daily, current_actions = run_current(context, policy)
    production_daily, production_actions = research_base.run_shell(
        context.harness,
        context.arrays,
        context.protocol,
        context.score,
        context.order,
        context.definition,
        "production_shell",
        round1.DEVELOPMENT_END,
        actions=True,
    )
    current_metrics = round1.evaluate_run(
        current_daily, current_actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    production_metrics = round1.evaluate_run(
        production_daily,
        production_actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
    )
    equivalent = {
        "fixed10": {
            key: bool(np.isclose(
                current_metrics[key], checkpoint["current_best_equalweight"][key],
                rtol=0.0, atol=1e-12,
            ))
            for key in (
                "cumulative_return", "cagr", "sharpe", "max_drawdown",
                "turnover_annualized", "average_invested_ratio",
            )
        },
        "production": {
            key: bool(np.isclose(
                production_metrics[key], checkpoint["production_baseline"][key],
                rtol=0.0, atol=1e-12,
            ))
            for key in (
                "cumulative_return", "cagr", "sharpe", "max_drawdown",
                "turnover_annualized", "average_invested_ratio",
            )
        },
    }
    if not all(equivalent["fixed10"].values()) or not all(
        equivalent["production"].values()
    ):
        raise RuntimeError("rolling robustness baseline drifted")
    frame = aligned_returns(current_daily, production_daily)
    quarters = calendar_period_summary(frame, "Q")
    years = calendar_period_summary(frame, "Y")
    result = {
        "status": "rolling_relative_robustness_complete_2026_not_opened",
        "rolling_windows": {
            str(window): rolling_summary(frame, window) for window in WINDOWS
        },
        "quarterly": quarters,
        "quarterly_outperformance_frequency": float(
            np.mean([row["excess_return"] > 0 for row in quarters])
        ),
        "annual": years,
        "annual_outperformance_frequency": float(
            np.mean([row["excess_return"] > 0 for row in years])
        ),
        "baseline_equivalence": equivalent,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "rolling_relative_robustness.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
