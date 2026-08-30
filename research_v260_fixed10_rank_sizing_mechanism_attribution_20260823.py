from __future__ import annotations

import copy
import functools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_mechanism_attribution_20260823"
)
TARGET_TOLERANCE = 1e-10


def target_bucket(target_pct: float) -> str:
    value = float(target_pct)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("active target weight must be finite and positive")
    if value > 0.10 + TARGET_TOLERANCE:
        return "above_10pct"
    if value < 0.10 - TARGET_TOLERANCE:
        return "below_10pct"
    return "equal_10pct"


def summarize_mechanism(
    observer_records: list[dict], actions: pd.DataFrame
) -> dict:
    required = {
        "buy_date", "action", "stock_code", "target_pct",
    }
    if not required.issubset(actions.columns):
        raise ValueError("action frame is missing mechanism attribution columns")
    actions_by_date = {
        str(date): group.reset_index(drop=True)
        for date, group in actions.groupby("buy_date", sort=False)
    }
    active_targets: dict[str, float] = {}
    bucket_rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    daily_comparison = []

    for record in observer_records:
        buy_date = str(record["buy_date"])
        day_returns = []
        day_notionals = []
        for mark in record["mark_records"]:
            code = str(mark["stock_code"])
            if code not in active_targets:
                raise RuntimeError(
                    "held position has no prior BUY target for mechanism attribution: "
                    f"{buy_date}/{code}"
                )
            prior_notional = float(mark["shares_before"]) * float(mark["prior_price"])
            mark_pnl = float(mark["mark_pnl"])
            if not np.isfinite(prior_notional) or prior_notional <= 0.0:
                raise ValueError("held prior notional must be finite and positive")
            if not np.isfinite(mark_pnl):
                raise ValueError("held mark PnL must be finite")
            mark_return = mark_pnl / prior_notional
            bucket_rows[target_bucket(active_targets[code])].append(
                (prior_notional, mark_pnl, mark_return)
            )
            day_notionals.append(prior_notional)
            day_returns.append(mark_return)

        if day_returns:
            notionals = np.asarray(day_notionals, dtype=np.float64)
            returns = np.asarray(day_returns, dtype=np.float64)
            actual_weighted = float(np.average(returns, weights=notionals))
            equal_held = float(returns.mean())
            daily_comparison.append(
                {
                    "date": buy_date,
                    "positions": int(len(returns)),
                    "actual_notional_weighted_return": actual_weighted,
                    "equal_notional_same_holdings_return": equal_held,
                }
            )

        for action in actions_by_date.get(buy_date, pd.DataFrame()).itertuples(
            index=False
        ):
            code = str(action.stock_code)
            if str(action.action) == "SELL":
                remaining_target = float(action.target_pct)
                if remaining_target <= TARGET_TOLERANCE:
                    active_targets.pop(code, None)
            elif str(action.action) == "BUY":
                if code not in active_targets:
                    active_targets[code] = float(action.target_pct)
            else:
                raise ValueError(f"unsupported action: {action.action}")

    bucket_summary = {}
    for bucket in ("above_10pct", "equal_10pct", "below_10pct"):
        values = bucket_rows.get(bucket, [])
        prior = np.asarray([item[0] for item in values], dtype=np.float64)
        pnl = np.asarray([item[1] for item in values], dtype=np.float64)
        returns = np.asarray([item[2] for item in values], dtype=np.float64)
        bucket_summary[bucket] = {
            "position_day_observations": int(len(values)),
            "prior_notional": float(prior.sum()) if len(prior) else 0.0,
            "gross_mark_pnl": float(pnl.sum()) if len(pnl) else 0.0,
            "pooled_mark_return": (
                float(pnl.sum() / prior.sum()) if len(prior) and prior.sum() > 0 else None
            ),
            "mean_position_day_return": float(returns.mean()) if len(returns) else None,
            "positive_position_day_fraction": (
                float((returns > 0.0).mean()) if len(returns) else None
            ),
        }

    daily = pd.DataFrame(daily_comparison)
    actual = daily["actual_notional_weighted_return"].to_numpy(dtype=np.float64)
    equal = daily["equal_notional_same_holdings_return"].to_numpy(dtype=np.float64)
    if (actual <= -1.0).any() or (equal <= -1.0).any():
        raise RuntimeError("same-holdings return decomposition is outside log domain")
    log_delta = np.log1p(actual) - np.log1p(equal)
    return {
        "bucket_summary": bucket_summary,
        "same_holdings_weight_effect": {
            "days": int(len(daily)),
            "actual_notional_weighted_cumulative": float(np.prod(1.0 + actual) - 1.0),
            "equal_notional_same_holdings_cumulative": float(np.prod(1.0 + equal) - 1.0),
            "compounded_relative_weight_effect": float(np.exp(log_delta.sum()) - 1.0),
            "positive_daily_weight_effect_fraction": float((log_delta > 0.0).mean()),
            "annualized_log_weight_effect": float(log_delta.mean() * 252.0),
        },
    }


def paired_path_attribution(
    candidate_records: list[dict], equalweight_records: list[dict]
) -> dict:
    candidate_by_date = {str(row["buy_date"]): row for row in candidate_records}
    equal_by_date = {str(row["buy_date"]): row for row in equalweight_records}
    if candidate_by_date.keys() != equal_by_date.keys():
        raise ValueError("paired observer dates do not align")
    totals = {
        "common_holdings_mark_delta": 0.0,
        "candidate_only_holdings_mark_contribution": 0.0,
        "minus_equalweight_only_holdings_mark_contribution": 0.0,
        "trade_cost_cash_rounding_residual_delta": 0.0,
    }
    candidate_returns = []
    equal_returns = []
    identical_holding_days = 0
    for date in candidate_by_date:
        candidate = candidate_by_date[date]
        equal = equal_by_date[date]
        candidate_previous = float(candidate["previous_equity"])
        equal_previous = float(equal["previous_equity"])
        if candidate_previous <= 0.0 or equal_previous <= 0.0:
            raise ValueError("previous equity must remain positive")
        candidate_return = (
            float(candidate["equity_after_trades"]) / candidate_previous - 1.0
        )
        equal_return = float(equal["equity_after_trades"]) / equal_previous - 1.0
        candidate_returns.append(candidate_return)
        equal_returns.append(equal_return)
        candidate_marks = {
            str(item["stock_code"]): float(item["mark_pnl"]) / candidate_previous
            for item in candidate["mark_records"]
        }
        equal_marks = {
            str(item["stock_code"]): float(item["mark_pnl"]) / equal_previous
            for item in equal["mark_records"]
        }
        common = candidate_marks.keys() & equal_marks.keys()
        candidate_only = candidate_marks.keys() - equal_marks.keys()
        equal_only = equal_marks.keys() - candidate_marks.keys()
        if not candidate_only and not equal_only:
            identical_holding_days += 1
        common_delta = sum(
            candidate_marks[code] - equal_marks[code] for code in common
        )
        candidate_only_contribution = sum(
            candidate_marks[code] for code in candidate_only
        )
        minus_equal_only_contribution = -sum(
            equal_marks[code] for code in equal_only
        )
        candidate_mark_total = sum(candidate_marks.values())
        equal_mark_total = sum(equal_marks.values())
        residual_delta = (
            candidate_return
            - candidate_mark_total
            - (equal_return - equal_mark_total)
        )
        daily_components = (
            common_delta
            + candidate_only_contribution
            + minus_equal_only_contribution
            + residual_delta
        )
        if not np.isclose(
            daily_components,
            candidate_return - equal_return,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("paired path attribution does not reconcile")
        totals["common_holdings_mark_delta"] += common_delta
        totals[
            "candidate_only_holdings_mark_contribution"
        ] += candidate_only_contribution
        totals[
            "minus_equalweight_only_holdings_mark_contribution"
        ] += minus_equal_only_contribution
        totals["trade_cost_cash_rounding_residual_delta"] += residual_delta

    candidate_array = np.asarray(candidate_returns, dtype=np.float64)
    equal_array = np.asarray(equal_returns, dtype=np.float64)
    arithmetic_delta = float((candidate_array - equal_array).sum())
    component_sum = float(sum(totals.values()))
    if not np.isclose(component_sum, arithmetic_delta, rtol=0.0, atol=1e-10):
        raise RuntimeError("aggregate paired path attribution does not reconcile")
    return {
        "days": int(len(candidate_array)),
        "identical_holding_set_days": int(identical_holding_days),
        "identical_holding_set_fraction": float(
            identical_holding_days / len(candidate_array)
        ),
        "arithmetic_daily_return_delta_sum": arithmetic_delta,
        "components": {key: float(value) for key, value in totals.items()},
        "component_sum": component_sum,
        "compounded_relative_return": float(
            np.prod(1.0 + candidate_array) / np.prod(1.0 + equal_array) - 1.0
        ),
    }


def run_with_observer(context, policy: dict, rank_sizing: bool):
    records: list[dict] = []
    extra_age, sell_priority = sizing.width_tools.build_overrides(context, policy)
    multipliers = (
        sizing.entry_rank_multipliers(
            context.order,
            sizing.TARGET_POSITIONS,
            sizing.TOP_WEIGHT_MULTIPLIER,
            sizing.BOTTOM_WEIGHT_MULTIPLIER,
        )
        if rank_sizing
        else None
    )
    simulator = functools.partial(
        sizing.runtime.simulate,
        position_observer=records.append,
        **(
            {"candidate_target_multiplier_override": multipliers}
            if multipliers is not None
            else {}
        ),
    )
    daily, actions = sizing.age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        sizing.BASELINE_COST,
        simulator=simulator,
        score_sell_priority_override=sell_priority,
        target_positions_override=sizing.TARGET_POSITIONS,
        target_gross_override=1.0,
    )
    return daily, actions, records


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered rank sizing mechanism attribution")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    daily, actions, records = run_with_observer(context, policy, True)
    equal_daily, equal_actions, equal_records = run_with_observer(
        context, policy, False
    )
    metrics = sizing.evaluate(daily, actions)
    equal_metrics = sizing.evaluate(equal_daily, equal_actions)
    for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown"):
        if not np.isclose(
            metrics[key], checkpoint["metrics_0_30pct"][key], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("mechanism replay drifted from frozen candidate")

    result = {
        "status": "rank_sizing_mechanism_attribution_complete_2026_not_opened",
        "role": "profit_source_diagnostic_not_a_selection_gate",
        "candidate_id": checkpoint["selected_candidate"],
        "candidate_metrics_0_30pct": metrics,
        "equalweight_metrics_0_30pct": equal_metrics,
        "attribution": summarize_mechanism(records, actions),
        "paired_candidate_vs_equalweight_path_attribution": paired_path_attribution(
            records, equal_records
        ),
        "interpretation_boundary": (
            "entry buckets remain fixed from first BUY through final liquidation; "
            "same-holdings comparison isolates realized notional weighting before "
            "trade costs; this is diagnostic and does not create a new strategy rule"
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "mechanism_attribution.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_id": result["candidate_id"],
                "same_holdings_weight_effect": result["attribution"][
                    "same_holdings_weight_effect"
                ],
                "validation_2026_opened": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
