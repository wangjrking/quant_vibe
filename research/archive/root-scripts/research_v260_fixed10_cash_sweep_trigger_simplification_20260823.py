from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

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
    "strategy_agent_v260_fixed10_cash_sweep_trigger_simplification_20260823"
)


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered cash-sweep trigger simplification")
    policy = checkpoint["selected_policy"]
    context = sizing.width_tools.harness.load_context(policy)
    cases = {}
    frames = {}
    for cost_name, cost in (
        ("0_30pct", sizing.BASELINE_COST),
        ("0_65pct", sizing.STRESS_COST),
    ):
        for trigger in ("any_trade", "buy_trade"):
            daily, actions = sweep.run_with_runtime(
                context,
                policy,
                cost,
                sweep=True,
                sweep_trigger=trigger,
            )
            cases.setdefault(cost_name, {})[trigger] = sizing.evaluate(daily, actions)
            frames[(cost_name, trigger)] = (daily, actions)
    replay = sweep.run_with_runtime(
        context,
        policy,
        sizing.BASELINE_COST,
        sweep=True,
        sweep_trigger="any_trade",
    )
    deterministic = all(
        actual.equals(expected)
        for actual, expected in zip(replay, frames[("0_30pct", "any_trade")])
    )
    if not deterministic:
        raise RuntimeError("cash-sweep trigger baseline replay drifted")
    deltas = {
        cost_name: sizing.checkpoint_tools.metric_delta(
            values["buy_trade"], values["any_trade"]
        )
        for cost_name, values in cases.items()
    }
    selected = (
        "buy_trade"
        if deltas["0_30pct"]["cumulative_return"] > 0.0
        else "any_trade"
    )
    result = {
        "status": "cash_sweep_trigger_simplification_complete_2026_not_opened",
        "single_ablation": "only trigger residual-cash sweep on a day with a buy",
        "selection_rule": "higher_pre2026_net_cumulative_return_at_0_30pct_cost",
        "cases": cases,
        "buy_trade_minus_any_trade": deltas,
        "selected_arm": selected,
        "parameter_grid_used": False,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_arm": selected,
                "baseline_return_delta": deltas["0_30pct"]["cumulative_return"],
                "stress_return_delta": deltas["0_65pct"]["cumulative_return"],
                "baseline_turnover_delta": deltas["0_30pct"]["turnover_annualized"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
