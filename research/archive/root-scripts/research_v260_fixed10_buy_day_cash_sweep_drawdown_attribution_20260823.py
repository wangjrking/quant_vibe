from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_cost_frontier_20260822 as production_compare
import research_v260_fixed10_drawdown_attribution_20260822 as drawdown
import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_rank_sizing_residual_cash_sweep_20260823 as sweep


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_buy_day_cash_sweep_drawdown_attribution_20260823"
)


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered current-candidate drawdown attribution")
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
    candidate_episode = drawdown.maximum_drawdown_episode(candidate_daily)
    production_episode = drawdown.maximum_drawdown_episode(production_daily)
    result = {
        "status": "current_candidate_drawdown_attribution_complete_2026_not_opened",
        "role": "risk_disclosure_only_no_strategy_gate",
        "candidate_id": checkpoint["selected_candidate"],
        "candidate_maximum_drawdown": candidate_episode,
        "production_maximum_drawdown": production_episode,
        "candidate_actions_in_maximum_drawdown": drawdown.actions_in_window(
            candidate_actions,
            candidate_episode["peak_date"],
            candidate_episode["trough_date"],
        ),
        "production_actions_in_maximum_drawdown": drawdown.actions_in_window(
            production_actions,
            production_episode["peak_date"],
            production_episode["trough_date"],
        ),
        "candidate_worst_windows": {
            str(window): drawdown.worst_compound_window(candidate_daily, window)
            for window in (20, 60)
        },
        "production_worst_windows": {
            str(window): drawdown.worst_compound_window(production_daily, window)
            for window in (20, 60)
        },
        "worst_relative_windows": {
            str(window): drawdown.relative_window(
                candidate_daily, production_daily, window
            )
            for window in (20, 60)
        },
        "daily_hashes": {
            "candidate": round1.frame_hash(candidate_daily),
            "production": round1.frame_hash(production_daily),
        },
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "drawdown_attribution.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
