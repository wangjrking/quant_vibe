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

import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_paired_block_bootstrap_20260822 as bootstrap
import research_v260_fixed10_rank_sizing_residual_cash_sweep_20260823 as sweep


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_residual_cash_sweep_robustness_20260823"
)
SEED = 2_602_026_082_3


def paired_log_diagnostics(joined: pd.DataFrame) -> dict:
    required = {"date", "current", "sweep"}
    if set(joined.columns) != required:
        raise ValueError("paired return frame columns are invalid")
    if joined.empty or joined["date"].duplicated().any():
        raise ValueError("paired return frame must contain unique dates")
    values = joined[["current", "sweep"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("paired returns are invalid")
    excess = np.log1p(values[:, 1]) - np.log1p(values[:, 0])
    dates = joined["date"].astype(str)
    years = dates.str[:4]
    annual = {
        str(year): {
            "log_excess": float(excess[years.to_numpy() == year].sum()),
            "relative_compound_excess": float(
                np.expm1(excess[years.to_numpy() == year].sum())
            ),
        }
        for year in sorted(years.unique())
    }
    rolling = {}
    for window in (126, 252):
        sums = pd.Series(excess).rolling(window).sum().dropna().to_numpy()
        rolling[str(window)] = {
            "window_count": int(len(sums)),
            "positive_fraction": float(np.mean(sums > 0.0)),
            "minimum_log_excess": float(np.min(sums)),
            "median_log_excess": float(np.median(sums)),
            "maximum_log_excess": float(np.max(sums)),
        }
    order = np.argsort(excess)[::-1]
    without_top = {}
    for count in (1, 5, 10, 20):
        remaining = float(excess.sum() - excess[order[:count]].sum())
        without_top[str(count)] = {
            "relative_compound_excess": float(np.expm1(remaining)),
            "removed_dates": dates.iloc[order[:count]].tolist(),
        }
    starts = {}
    for offset in (0, 20, 60, 126):
        starts[str(offset)] = {
            "start_date": str(dates.iloc[offset]),
            "relative_compound_excess": float(np.expm1(excess[offset:].sum())),
        }
    return {
        "total_log_excess": float(excess.sum()),
        "total_relative_compound_excess": float(np.expm1(excess.sum())),
        "positive_excess_day_fraction": float(np.mean(excess > 0.0)),
        "annual": annual,
        "positive_year_count": int(
            sum(item["relative_compound_excess"] > 0.0 for item in annual.values())
        ),
        "rolling_log_excess": rolling,
        "without_top_positive_excess_days": without_top,
        "fixed_start_offsets": starts,
    }


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered cash-sweep robustness")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    current_daily, current_actions = sizing.run_case(
        context,
        policy,
        sizing.BASELINE_COST,
        sizing.TOP_WEIGHT_MULTIPLIER,
        sizing.BOTTOM_WEIGHT_MULTIPLIER,
    )
    sweep_daily, sweep_actions = sweep.run_with_runtime(
        context, policy, sizing.BASELINE_COST, sweep=True
    )
    joined = (
        current_daily[["date", "return"]]
        .rename(columns={"return": "current"})
        .merge(
            sweep_daily[["date", "return"]].rename(columns={"return": "sweep"}),
            on="date",
            validate="one_to_one",
        )
    )
    joined["date"] = joined["date"].astype(str)
    joined = joined.loc[
        (joined["date"] >= sizing.research_base.FIRST_BUY)
        & (joined["date"] <= round1.DEVELOPMENT_END)
    ].reset_index(drop=True)
    if len(joined) != int(checkpoint["metrics_0_30pct"]["days"]):
        raise RuntimeError("cash-sweep robustness date alignment drifted")

    current_values = joined["current"].to_numpy(dtype=np.float64)
    sweep_values = joined["sweep"].to_numpy(dtype=np.float64)
    diagnostics = paired_log_diagnostics(joined)
    block_bootstrap = {
        str(block): bootstrap.paired_bootstrap(
            sweep_values,
            current_values,
            block,
            SEED + block,
        )
        for block in (5, 20, 60)
    }
    result = {
        "status": "residual_cash_sweep_robustness_complete_2026_not_opened",
        "role": "profit_and_concentration_diagnostic_not_a_new_rejection_contract",
        "paired_diagnostics": diagnostics,
        "paired_block_bootstrap": block_bootstrap,
        "metrics": {
            "current": sizing.evaluate(current_daily, current_actions),
            "residual_cash_sweep": sizing.evaluate(sweep_daily, sweep_actions),
        },
        "interpretation": (
            "use the breadth and concentration evidence to judge whether the extra "
            "pre-2026 profit is credible; no annual, rolling or bootstrap statistic "
            "is converted into a new hard strategy gate"
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "robustness.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "relative_excess": diagnostics["total_relative_compound_excess"],
                "positive_year_count": diagnostics["positive_year_count"],
                "rolling252_positive_fraction": diagnostics[
                    "rolling_log_excess"
                ]["252"]["positive_fraction"],
                "bootstrap_probability_higher": {
                    block: item["probability_annualized_log_return_higher"]
                    for block, item in block_bootstrap.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
