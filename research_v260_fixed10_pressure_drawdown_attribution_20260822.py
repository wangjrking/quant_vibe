from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_drawdown_attribution_20260822 as drawdown
import research_v260_fixed10_drawdown_position_attribution_20260822 as position
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_drawdown_attribution_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def run_with_observer(policy: dict):
    context = harness.load_context(policy)
    observations: list[dict] = []
    daily, actions = round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        context.order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=context.active,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=context.empty_block,
        maintenance_buy_block_mask_override=(
            ~np.isfinite(context.score)
            | (
                context.score
                < float(policy["maintenance_topup_requires_score"])
            )
        ),
        score_sell_pressure_trigger_override=policy["score_sell_pressure_trigger"],
        score_sell_pressure_limit_override=policy["score_sell_pressure_limit"],
        score_sell_pressure_confirmation_days_override=policy[
            "score_sell_pressure_confirmation_days"
        ],
        position_observer=observations.append,
    )
    return context, daily, actions, observations


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered pressure drawdown attribution")
    policy = checkpoint["selected_policy"]
    context, daily, actions, observations = run_with_observer(policy)
    episode = drawdown.maximum_drawdown_episode(daily)
    aggregate = position.aggregate_episode(
        observations, episode["peak_date"], episode["trough_date"]
    )
    metrics = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    checkpoint_equivalence = {
        key: bool(
            np.isclose(
                metrics[key],
                checkpoint["current_best_equalweight"][key],
                rtol=0.0,
                atol=1e-12,
            )
        )
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("pressure drawdown attribution drifted")
    _, _, _, repeat_observations = run_with_observer(policy)
    deterministic = position.observation_hash(observations) == position.observation_hash(
        repeat_observations
    )
    if not deterministic:
        raise RuntimeError("pressure drawdown attribution replay failed")

    result = {
        "status": "pressure_drawdown_diagnostic_complete_2026_not_opened",
        "maximum_drawdown_episode": episode,
        "position_attribution": aggregate,
        "interpretation": (
            "broad-based losses call for portfolio-level defense; concentrated losses "
            "call for a position-level rule"
        ),
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "drawdown_attribution.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
