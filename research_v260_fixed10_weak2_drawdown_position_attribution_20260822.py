from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_drawdown_attribution_20260822 as drawdown
import research_v260_fixed10_drawdown_position_attribution_20260822 as attribution
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_weak2_drawdown_position_attribution_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def run_with_observer(context, policy: dict):
    threshold = policy.get("maintenance_topup_requires_score")
    maintenance_block = (
        context.empty_block
        if threshold is None
        else ~np.isfinite(context.score) | (context.score < float(threshold))
    )
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_min_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
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
        portfolio_rebalance_active_override=cadence.rebalance_schedule(
            len(context.arrays["dates"]),
            int(policy["portfolio_rebalance_interval_days"]),
        ),
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=context.empty_block,
        maintenance_buy_block_mask_override=maintenance_block,
        score_sell_pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_pressure_limit_override=policy["score_sell_pressure_limit"],
        score_sell_pressure_confirmation_days_override=policy.get(
            "score_sell_pressure_confirmation_days", 1
        ),
        score_sell_pressure_extra_min_age_override=extra_min_age,
        score_sell_priority_override=priority_matrix,
        position_observer=observations.append,
    )
    return daily, actions, observations, volatility_rank, strong, confirmed_weak


def episode_action_diagnostics(
    actions,
    context,
    volatility_rank: np.ndarray,
    strong_market: np.ndarray,
    confirmed_weak: np.ndarray,
    peak_date: str,
    trough_date: str,
) -> dict:
    stock_index = {
        str(stock_code): index
        for index, stock_code in enumerate(context.arrays["stocks"])
    }
    date_index = {
        str(trade_date): index
        for index, trade_date in enumerate(context.arrays["dates"])
    }
    episode = actions[
        (actions["buy_date"].astype(str) > peak_date)
        & (actions["buy_date"].astype(str) <= trough_date)
    ]
    rows = []
    for action in episode.itertuples(index=False):
        signal_date = str(action.signal_date)
        stock_code = str(action.stock_code)
        t = date_index[signal_date]
        idx = stock_index[stock_code]
        rows.append(
            {
                "signal_date": signal_date,
                "buy_date": str(action.buy_date),
                "action": str(action.action),
                "stock_code": stock_code,
                "score": float(context.score[t, idx]),
                "volatility_percentile": float(volatility_rank[t, idx]),
                "strong_market": bool(strong_market[t]),
                "confirmed_weak_market": bool(confirmed_weak[t]),
            }
        )
    buys = [row for row in rows if row["action"] == "BUY"]
    sells = [row for row in rows if row["action"] == "SELL"]

    def summarize(items: list[dict]) -> dict:
        if not items:
            return {"count": 0}
        return {
            "count": len(items),
            "median_score": float(np.median([item["score"] for item in items])),
            "median_volatility_percentile": float(
                np.median([item["volatility_percentile"] for item in items])
            ),
            "confirmed_weak_count": int(
                sum(item["confirmed_weak_market"] for item in items)
            ),
        }

    return {
        "buy_summary": summarize(buys),
        "sell_summary": summarize(sells),
        "buys": buys,
        "sells": sells,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for attribution")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    daily, actions, observations, volatility_rank, strong, confirmed_weak = (
        run_with_observer(context, policy)
    )
    metrics = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(metrics[key], expected[key], rtol=0.0, atol=1e-12))
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
        raise RuntimeError("weak2 checkpoint drifted during attribution")
    episode = drawdown.maximum_drawdown_episode(daily)
    aggregate = attribution.aggregate_episode(
        observations, episode["peak_date"], episode["trough_date"]
    )
    action_diagnostics = episode_action_diagnostics(
        actions,
        context,
        volatility_rank,
        strong,
        confirmed_weak,
        episode["peak_date"],
        episode["trough_date"],
    )
    _, _, repeated, _, _, _ = run_with_observer(context, policy)
    deterministic = (
        attribution.observation_hash(observations)
        == attribution.observation_hash(repeated)
    )
    if not deterministic:
        raise RuntimeError("weak2 position attribution replay failed")

    result = {
        "status": "weak2_drawdown_attribution_complete_2026_not_opened",
        "maximum_drawdown_episode": episode,
        "position_attribution": aggregate,
        "episode_action_diagnostics": action_diagnostics,
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "position_attribution.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
