from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_cost_frontier_20260822 as production_compare
import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_paired_block_bootstrap_20260822 as bootstrap
import research_v260_fixed10_rank_sizing_residual_cash_sweep_20260823 as sweep
import research_v260_fixed10_residual_cash_sweep_robustness_20260823 as diagnostics


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_buy_day_cash_sweep_vs_production_robustness_20260823"
)
SEED = 2_602_026_082_4


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered candidate-production robustness")
    policy = checkpoint["selected_policy"]
    context = sizing.width_tools.harness.load_context(policy)
    candidate_daily, candidate_actions = sweep.run_with_runtime(
        context,
        policy,
        sizing.BASELINE_COST,
        sweep=True,
        rank_sizing=True,
        sweep_trigger="buy_trade",
        sweep_recipient="most_underweight",
    )
    production_daily, production_actions = production_compare.run_production(
        context, sizing.BASELINE_COST
    )
    paired = (
        production_daily[["date", "return"]]
        .rename(columns={"return": "current"})
        .merge(
            candidate_daily[["date", "return"]].rename(columns={"return": "sweep"}),
            on="date",
            validate="one_to_one",
        )
    )
    paired["date"] = paired["date"].astype(str)
    paired_result = diagnostics.paired_log_diagnostics(paired)
    candidate_returns = paired["sweep"].to_numpy(dtype=np.float64)
    production_returns = paired["current"].to_numpy(dtype=np.float64)
    block_result = {
        str(block): bootstrap.paired_bootstrap(
            candidate_returns,
            production_returns,
            block,
            SEED + block,
        )
        for block in (5, 20, 60)
    }
    result = {
        "status": "current_candidate_vs_production_robustness_complete_2026_not_opened",
        "role": "profit_breadth_disclosure_only_no_strategy_gate",
        "candidate_id": checkpoint["selected_candidate"],
        "paired_diagnostics": paired_result,
        "paired_block_bootstrap": block_result,
        "candidate_metrics": sizing.evaluate(candidate_daily, candidate_actions),
        "production_metrics": round1.evaluate_run(
            production_daily,
            production_actions,
            sizing.research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
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
                "relative_compound_excess": paired_result[
                    "total_relative_compound_excess"
                ],
                "positive_year_count": paired_result["positive_year_count"],
                "rolling252_positive_fraction": paired_result[
                    "rolling_log_excess"
                ]["252"]["positive_fraction"],
                "bootstrap_probability_higher": {
                    key: value["probability_annualized_log_return_higher"]
                    for key, value in block_result.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
