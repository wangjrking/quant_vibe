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
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_event_entry_block_robustness_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
BLOCK_RATES = (0.01, 0.05, 0.10)
SEEDS = tuple(range(10))


def decision_components(context, policy: dict):
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    sell_priority = priority.sell_priority_matrix(
        context.score, volatility_rank, ~confirmed_weak, 0.05
    )
    return extra_age, sell_priority


def baseline_buy_keys(actions) -> list[tuple[str, str]]:
    buys = actions.loc[actions["action"].eq("BUY"), ["signal_date", "stock_code"]]
    keys = [tuple(map(str, row)) for row in buys.itertuples(index=False, name=None)]
    if len(keys) != len(set(keys)):
        raise RuntimeError("baseline buy keys are not unique")
    return keys


def sampled_block_mask(context, keys, rate: float, seed: int) -> np.ndarray:
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    count = max(1, int(round(len(keys) * float(rate))))
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(len(keys), size=count, replace=False)
    dates = {str(value): index for index, value in enumerate(context.arrays["dates"])}
    stocks = {
        str(value): index for index, value in enumerate(context.arrays["stocks"])
    }
    mask = np.zeros(context.score.shape, dtype=np.bool_)
    for item in selected:
        date, stock = keys[int(item)]
        if date not in dates or stock not in stocks:
            raise KeyError(f"baseline buy key not found: {date}/{stock}")
        mask[dates[date], stocks[stock]] = True
    if int(mask.sum()) != count:
        raise RuntimeError("sampled block mask count drifted")
    return mask


def distribution(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered event-entry block robustness")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    extra_age, sell_priority = decision_components(context, policy)
    baseline, baseline_daily, baseline_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        score_sell_priority_override=sell_priority,
    )
    expected = checkpoint["current_best_equalweight"]
    baseline_equivalence = {
        key: bool(
            np.isclose(
                baseline["metrics_0_30pct"][key],
                expected[key],
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
    if not all(baseline_equivalence.values()):
        raise RuntimeError("event-entry block baseline drifted")
    keys = baseline_buy_keys(baseline_actions)
    production = checkpoint["production_baseline"]

    runs = {}
    summary = {}
    for rate in BLOCK_RATES:
        rate_id = f"{rate:.2f}"
        runs[rate_id] = {}
        for seed in SEEDS:
            block = sampled_block_mask(context, keys, rate, seed)
            result, daily, actions = age_guard.run_policy(
                context,
                policy,
                extra_age,
                entry_block_mask_override=block,
                maintenance_buy_block_mask_override=block,
                score_sell_priority_override=sell_priority,
            )
            metrics = result["metrics_0_30pct"]
            runs[rate_id][str(seed)] = {
                "blocked_baseline_buy_keys": int(block.sum()),
                "metrics_0_30pct": metrics,
                "stress_cumulative_return": result["metrics_0_65pct"][
                    "cumulative_return"
                ],
                "all_calendar_years_positive": all(
                    value > 0.0 for value in metrics["annual_returns"].values()
                ),
                "full_10_positions": metrics["full_10_position_ratio"] == 1.0,
                "daily_hash": round1.frame_hash(daily),
                "actions_hash": round1.frame_hash(actions),
            }
        rate_runs = list(runs[rate_id].values())
        summary[rate_id] = {
            "blocked_keys": rate_runs[0]["blocked_baseline_buy_keys"],
            "cumulative_return": distribution(
                item["metrics_0_30pct"]["cumulative_return"] for item in rate_runs
            ),
            "sharpe": distribution(
                item["metrics_0_30pct"]["sharpe"] for item in rate_runs
            ),
            "max_drawdown": distribution(
                item["metrics_0_30pct"]["max_drawdown"] for item in rate_runs
            ),
            "stress_cumulative_return": distribution(
                item["stress_cumulative_return"] for item in rate_runs
            ),
            "all_years_positive_fraction": float(
                np.mean([item["all_calendar_years_positive"] for item in rate_runs])
            ),
            "full_10_positions_fraction": float(
                np.mean([item["full_10_positions"] for item in rate_runs])
            ),
            "beats_production_cumulative_fraction": float(
                np.mean(
                    [
                        item["metrics_0_30pct"]["cumulative_return"]
                        > production["cumulative_return"]
                        for item in rate_runs
                    ]
                )
            ),
        }

    repeat_mask = sampled_block_mask(context, keys, 0.05, 0)
    _, repeat_daily, repeat_actions = age_guard.run_policy(
        context,
        policy,
        extra_age,
        entry_block_mask_override=repeat_mask,
        maintenance_buy_block_mask_override=repeat_mask,
        score_sell_priority_override=sell_priority,
    )
    deterministic = {
        "daily": runs["0.05"]["0"]["daily_hash"]
        == round1.frame_hash(repeat_daily),
        "actions": runs["0.05"]["0"]["actions_hash"]
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("event-entry block replay failed")

    payload = {
        "status": "event_entry_block_robustness_complete_2026_not_opened",
        "method": (
            "deterministically sample and block baseline buy stock-date keys, then let "
            "the unchanged refill logic choose alternatives; probes are not selectable"
        ),
        "baseline_buy_key_count": len(keys),
        "block_rates": list(BLOCK_RATES),
        "seeds": list(SEEDS),
        "summary": summary,
        "runs": runs,
        "baseline_equivalence": baseline_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "event_entry_block_robustness.json", payload)
    print(
        json.dumps(
            {"status": payload["status"], "summary": summary},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
