# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.main.research_v260_risk_event_overlay import (
    entry_ranked_without_events,
    maintenance_buy_allowed,
)

from quant.main.strategy_library.production.prod_v260_10d_regime_warmup_all4key_v20260724.production_code.v260_all4key_runtime import (
    production_v260_active_l4_10d_smoothing_refine_v95_20260722 as v95,
)
from quant.main.strategy_library.production.prod_v260_10d_regime_warmup_all4key_v20260724.production_code.v260_all4key_runtime import (
    production_v260_active_l4_relaxed_universe_v86_20260722 as v86,
)
from quant.main.strategy_library.production.prod_v260_10d_regime_warmup_all4key_v20260724.production_code.v260_all4key_runtime import (
    production_v260_active_l4_robust_objective_v56_20260721 as robust,
)
from quant.main.strategy_library.production.prod_v260_10d_regime_warmup_all4key_v20260724.production_code.v260_all4key_runtime import (
    production_v260_preregistered_active_l4_rank_rotation_20260721 as core,
)


OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_breadth_exit_v109_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"
ACTION_COLUMNS = ["signal_date", "buy_date", "action", "stock_code", "target_pct", "execution_open_raw"]


def residual_cash_sweep_triggered(
    trigger: str, trades: int, bought_indices_today: set[int]
) -> bool:
    if trigger == "any_trade":
        return trades > 0
    if trigger == "buy_trade":
        return bool(bought_indices_today)
    raise ValueError("cash sweep trigger is invalid")


def residual_cash_sweep_plan(
    *,
    stocks: np.ndarray,
    shares: dict[int, float],
    raw_opens: np.ndarray,
    mark_prices: np.ndarray,
    eligible_indices: set[int],
    cash: float,
    target_pct: float,
    slippage_ratio: float,
    max_additions: int | None = None,
    recipient_mode: str = "most_underweight",
    priority_scores: np.ndarray | None = None,
) -> tuple[float, dict[int, float]]:
    """Spend affordable residual cash on the most underweight held names."""
    if raw_opens.ndim != 1 or mark_prices.shape != raw_opens.shape:
        raise ValueError("cash sweep prices must be aligned vectors")
    if len(stocks) != len(raw_opens):
        raise ValueError("cash sweep stocks and prices must be aligned")
    if (
        not np.isfinite(cash)
        or cash < 0.0
        or not np.isfinite(target_pct)
        or target_pct <= 0.0
        or not np.isfinite(slippage_ratio)
        or not 0.0 <= slippage_ratio < 1.0
    ):
        raise ValueError("cash sweep scalar inputs are invalid")
    if any(idx not in shares for idx in eligible_indices):
        raise ValueError("cash sweep may only top up existing holdings")
    if max_additions is not None and max_additions < 1:
        raise ValueError("cash sweep maximum additions must be positive")
    if recipient_mode not in {"most_underweight", "highest_score"}:
        raise ValueError("cash sweep recipient mode is invalid")
    if recipient_mode == "highest_score":
        if priority_scores is None or priority_scores.shape != raw_opens.shape:
            raise ValueError("cash sweep priority scores must be aligned")
        if any(not np.isfinite(priority_scores[idx]) for idx in eligible_indices):
            raise ValueError("cash sweep eligible priority scores must be finite")
    if any(
        idx < 0
        or idx >= len(stocks)
        or not np.isfinite(raw_opens[idx])
        or raw_opens[idx] <= 0.0
        or not np.isfinite(mark_prices[idx])
        or mark_prices[idx] <= 0.0
        for idx in eligible_indices
    ):
        raise ValueError("cash sweep eligible prices are invalid")

    working_shares = {idx: float(quantity) for idx, quantity in shares.items()}
    if any(not np.isfinite(value) or value < 0.0 for value in working_shares.values()):
        raise ValueError("cash sweep shares are invalid")
    additions: dict[int, float] = {}
    unavailable: set[int] = set()
    remaining_cash = float(cash)
    while True:
        if max_additions is not None and len(additions) >= max_additions:
            break
        marked_equity = remaining_cash + sum(
            working_shares[idx] * float(mark_prices[idx]) for idx in working_shares
        )
        target_value = marked_equity * target_pct
        if recipient_mode == "highest_score":
            candidates = sorted(
                eligible_indices - unavailable,
                key=lambda idx: (-float(priority_scores[idx]), str(stocks[idx])),
            )
        else:
            candidates = sorted(
                eligible_indices - unavailable,
                key=lambda idx: (
                    working_shares[idx] * float(raw_opens[idx]) - target_value,
                    str(stocks[idx]),
                ),
            )
        if not candidates:
            break
        idx = int(candidates[0])
        shortfall = target_value - working_shares[idx] * float(raw_opens[idx])
        execution_price = float(raw_opens[idx]) * (1.0 + slippage_ratio)
        budget = min(max(shortfall, 0.0), remaining_cash)
        quantity = math.floor(budget / (execution_price * 1.0003) / 100.0) * 100.0
        spend = quantity * execution_price * 1.0003
        if quantity < 100.0 or spend > remaining_cash + 1e-6:
            unavailable.add(idx)
            continue
        remaining_cash -= spend
        working_shares[idx] += quantity
        additions[idx] = additions.get(idx, 0.0) + quantity
    if remaining_cash < -1e-6:
        raise RuntimeError("cash sweep produced negative cash")
    return max(remaining_cash, 0.0), additions


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v109_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def entry_open_gap_allowed(opens, pre_close, maximum_gap=None, minimum_gap=None):
    open_values = np.asarray(opens, dtype=np.float64)
    prior_close = np.asarray(pre_close, dtype=np.float64)
    if open_values.shape != prior_close.shape:
        raise ValueError("entry open and pre-close shapes do not align")
    if maximum_gap is None and minimum_gap is None:
        return np.ones(open_values.shape, dtype=np.bool_)
    valid = (
        np.isfinite(open_values)
        & np.isfinite(prior_close)
        & (open_values > 0)
        & (prior_close > 0)
    )
    result = np.zeros(open_values.shape, dtype=np.bool_)
    gap = open_values[valid] / prior_close[valid] - 1.0
    allowed = np.ones(gap.shape, dtype=np.bool_)
    if maximum_gap is not None:
        allowed &= gap <= float(maximum_gap) + 1e-12
    if minimum_gap is not None:
        allowed &= gap >= float(minimum_gap) - 1e-12
    result[valid] = allowed
    return result


def resolve_score_sell_limit(
    base_limit: int,
    candidate_count: int,
    pressure_trigger: int | None = None,
    pressure_limit: int | None = None,
    pressure_streak: int = 1,
    confirmation_days: int = 1,
) -> int:
    limit = max(int(base_limit), 0)
    count = max(int(candidate_count), 0)
    if pressure_trigger is None and pressure_limit is None:
        return limit
    if pressure_trigger is None or pressure_limit is None:
        raise ValueError("pressure trigger and limit must be provided together")
    trigger = max(int(pressure_trigger), 1)
    elevated = max(int(pressure_limit), limit)
    confirmed = max(int(pressure_streak), 0) >= max(int(confirmation_days), 1)
    return elevated if count >= trigger and confirmed else limit


def apply_pressure_cooldown(
    resolved_limit: int,
    base_limit: int,
    current_index: int,
    last_extra_sell_index: int,
    cooldown_days: int,
) -> int:
    cooldown = max(int(cooldown_days), 0)
    if cooldown and int(current_index) - int(last_extra_sell_index) <= cooldown:
        return max(int(base_limit), 0)
    return max(int(resolved_limit), 0)


def apply_pressure_replacement_quality_gate(
    resolved_limit: int,
    base_limit: int,
    qualified_replacement_count: int,
) -> int:
    """Allow an elevated exit budget only when every sale has a qualified refill."""
    resolved = max(int(resolved_limit), 0)
    base = max(int(base_limit), 0)
    if resolved <= base:
        return resolved
    return resolved if max(int(qualified_replacement_count), 0) >= resolved else base


def pressure_sells_with_extra_min_age(
    ordered_score_sells,
    limit: int,
    base_limit: int,
    entry_index: dict[int, int],
    current_index: int,
    extra_min_age: int | None,
):
    if extra_min_age is None or limit <= base_limit:
        return list(ordered_score_sells[:limit])
    base = list(ordered_score_sells[:base_limit])
    eligible_extras = [
        idx
        for idx in ordered_score_sells[base_limit:]
        if current_index - entry_index[idx] >= int(extra_min_age)
    ]
    return (base + eligible_extras)[:limit]


def order_score_sells(score_sell_candidates, exit_scores, stock_codes, priority=None):
    """Order eligible score exits; lower priority values leave first."""
    scores = np.asarray(exit_scores, dtype=np.float64)
    codes = np.asarray(stock_codes).astype(str)
    priorities = scores if priority is None else np.asarray(priority, dtype=np.float64)
    if scores.shape != codes.shape or priorities.shape != scores.shape:
        raise ValueError("score-sell priority inputs do not align")
    return sorted(
        (int(idx) for idx in score_sell_candidates),
        key=lambda item: (
            float(priorities[item]) if np.isfinite(priorities[item]) else -np.inf,
            float(scores[item]) if np.isfinite(scores[item]) else -np.inf,
            codes[item],
        ),
    )


def pairwise_supported_score_sells(
    score_sell_candidates,
    replacement_candidates,
    exit_scores,
    replacement_scores,
    required_advantages,
    limit,
    reserved_replacements=0,
):
    ordered_sells = sorted(
        (int(idx) for idx in score_sell_candidates),
        key=lambda item: (
            float(exit_scores[item]) if np.isfinite(exit_scores[item]) else -np.inf,
            item,
        ),
    )
    replacements = [int(idx) for idx in replacement_candidates]
    start = max(int(reserved_replacements), 0)
    supported = []
    for sell_idx, replacement_idx in zip(
        ordered_sells[: max(int(limit), 0)], replacements[start:]
    ):
        current = float(exit_scores[sell_idx])
        replacement = float(replacement_scores[replacement_idx])
        required = float(required_advantages[sell_idx])
        if (
            not np.isfinite(current)
            or not np.isfinite(replacement)
            or replacement - current < required
        ):
            break
        supported.append(sell_idx)
    return supported


def simulate(
    arrays,
    score,
    order,
    definition,
    protocol,
    end_date,
    start=None,
    record_actions=False,
    breadth_count_override=None,
    gross_target_override=None,
    daily_exit_candidates=False,
    daily_replacement=False,
    rank_weight_power=0.0,
    rebalance_offset=0,
    rebalance_active_override=None,
    selection_mask_override=None,
    target_cohorts_override=None,
    target_pct_override=None,
    min_hold_days_override=None,
    candidate_target_multiplier_override=None,
    candidate_quantity_allocator_override=None,
    maintenance_target_multiplier_override=None,
    residual_cash_sweep_override=False,
    residual_cash_sweep_max_orders_override=None,
    residual_cash_sweep_trigger_override="any_trade",
    residual_cash_sweep_recipient_override="most_underweight",
    sell_confirmation_days_override=1,
    sell_score_below_override=None,
    sell_score_below_age_bands_override=None,
    replacement_advantage_override=None,
    replacement_advantage_age_bands_override=None,
    max_positions_override=None,
    max_positions_schedule_override=None,
    entry_slots_schedule_override=None,
    max_daily_score_sells_override=None,
    score_sell_pressure_trigger_override=None,
    score_sell_pressure_limit_override=None,
    score_sell_pressure_confirmation_days_override=1,
    score_sell_pressure_cooldown_days_override=0,
    score_sell_pressure_min_replacement_score_override=None,
    score_sell_pressure_extra_min_age_override=None,
    score_sell_priority_override=None,
    score_sell_pressure_observer=None,
    pairwise_replacement_guard_override=False,
    refill_after_sells_override=False,
    max_refill_buys_override=None,
    split_candidate_target_override=False,
    reentry_cooldown_days_override=0,
    max_hold_renewal_policy_override="none",
    min_entry_top_gap_override=0.0,
    max_hold_renewal_score_override=None,
    absolute_max_hold_days_override=None,
    portfolio_rebalance_active_override=None,
    portfolio_rebalance_min_weight_deviation_override=0.0,
    portfolio_rebalance_overweight_deviation_override=None,
    portfolio_rebalance_underweight_deviation_override=None,
    entry_block_mask_override=None,
    maintenance_buy_block_mask_override=None,
    score_exit_confirmation_mask_override=None,
    exit_score_override=None,
    price_peak_drawdown_exit_override=None,
    max_entry_open_gap_override=None,
    min_entry_open_gap_override=None,
    entry_price_loss_exit_override=None,
    position_observer=None,
    defer_sell_on_limit_up_override=False,
):
    rebalance_every = int(protocol["fixed_strategy"]["rebalance_every"])
    prepared = v86.stagger(arrays, rebalance_every)
    if rebalance_active_override is not None:
        prepared["signal_clean"] = arrays["signal_clean"].copy()
        prepared["signal_clean"] &= np.asarray(rebalance_active_override, dtype=np.bool_)[:, None]
    elif rebalance_offset:
        prepared["signal_clean"] = arrays["signal_clean"].copy()
        active = np.zeros(len(arrays["dates"]), dtype=np.bool_)
        active[int(rebalance_offset) :: rebalance_every] = True
        prepared["signal_clean"] &= active[:, None]
    dates, stocks = prepared["dates"].astype(str), prepared["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    cash = float(protocol["execution"]["initial_cash"])
    previous_equity = cash
    shares, entry_index, original_entry_index, entry_raw_price = {}, {}, {}, {}
    last_price, sell_streak, last_sell_index, peak_signal_close = {}, {}, {}, {}
    score_sell_pressure_streak = 0
    last_pressure_extra_sell_index = -10**9
    rows, actions = [], []
    top_n = int(protocol["fixed_strategy"]["top_n_per_rebalance"])
    max_hold = int(protocol["fixed_strategy"]["max_hold_days"])
    cohorts = math.ceil(max_hold / rebalance_every)
    slip = float(protocol["execution"]["fixed_slippage_ratio"])
    event_shape = (len(dates), len(stocks))
    entry_block = (
        np.zeros(event_shape, dtype=np.bool_)
        if entry_block_mask_override is None
        else np.asarray(entry_block_mask_override, dtype=np.bool_)
    )
    maintenance_block = (
        entry_block
        if maintenance_buy_block_mask_override is None
        else np.asarray(maintenance_buy_block_mask_override, dtype=np.bool_)
    )
    score_exit_confirmation = (
        np.ones(event_shape, dtype=np.bool_)
        if score_exit_confirmation_mask_override is None
        else np.asarray(score_exit_confirmation_mask_override, dtype=np.bool_)
    )
    exit_score = (
        score
        if exit_score_override is None
        else np.asarray(exit_score_override, dtype=np.float64)
    )
    score_sell_priority = (
        exit_score
        if score_sell_priority_override is None
        else np.asarray(score_sell_priority_override, dtype=np.float64)
    )
    if (
        entry_block.shape != event_shape
        or maintenance_block.shape != event_shape
        or score_exit_confirmation.shape != event_shape
        or exit_score.shape != event_shape
        or score_sell_priority.shape != event_shape
    ):
        raise ValueError("risk-event block mask shape mismatch")
    exit_order = (
        order
        if exit_score_override is None
        else np.argsort(
            -np.nan_to_num(exit_score, nan=-np.inf), axis=1, kind="stable"
        )
    )

    for t, signal_date in enumerate(dates[:-1]):
        if signal_date > end_date:
            break
        buy_date = str(dates[t + 1])
        if start and buy_date < start:
            continue
        opens, pre_close = prepared["buy_open"][t], prepared["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        mark_records = []
        for idx in list(shares):
            prior_price = float(last_price.get(idx, 0.0))
            current_price = float(opens[idx]) if valid_open[idx] else prior_price
            mark_records.append(
                {
                    "stock_code": stocks[idx],
                    "shares_before": float(shares[idx]),
                    "prior_price": prior_price,
                    "current_price": current_price,
                    "mark_pnl": float(shares[idx]) * (current_price - prior_price),
                    "current_open_available": bool(valid_open[idx]),
                }
            )
            if valid_open[idx]:
                last_price[idx] = current_price
        equity_before = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)

        common_clean = prepared["buy_clean"][t] & valid_open & np.isfinite(score[t])
        common_clean &= prepared["listed_days"][t] >= int(protocol["fixed_universe"]["listed_days_min"])
        common_clean &= np.isfinite(prepared["amount"][t]) & (prepared["amount"][t] >= int(protocol["fixed_universe"]["amount_min"]))
        common_clean &= np.isfinite(prepared["total_mv"][t]) & (prepared["total_mv"][t] >= int(protocol["fixed_universe"]["mv_min"]))
        common_clean &= np.isfinite(prepared["turnover_rate"][t]) & (prepared["turnover_rate"][t] >= 0) & (prepared["turnover_rate"][t] <= float(protocol["fixed_universe"]["turnover_max"]))
        common_clean &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        if selection_mask_override is not None:
            common_clean &= np.asarray(selection_mask_override[t], dtype=np.bool_)
        clean = prepared["signal_clean"][t] & common_clean
        if (
            max_entry_open_gap_override is not None
            or min_entry_open_gap_override is not None
        ):
            clean &= entry_open_gap_allowed(
                opens,
                pre_close,
                maximum_gap=max_entry_open_gap_override,
                minimum_gap=min_entry_open_gap_override,
            )
        ranked = [int(idx) for idx in order[t] if clean[int(idx)]]
        entry_ranked = entry_ranked_without_events(ranked, entry_block[t])
        min_entry_top_gap = max(float(min_entry_top_gap_override), 0.0)
        if min_entry_top_gap > 0:
            if (
                len(ranked) < 2
                or float(score[t, ranked[0]] - score[t, ranked[1]])
                < min_entry_top_gap
            ):
                entry_ranked = []
        if daily_exit_candidates:
            exit_clean = arrays["signal_clean"][t] & common_clean
            ranked_exit = [
                int(idx) for idx in exit_order[t] if exit_clean[int(idx)]
            ]
        elif exit_score_override is not None:
            ranked_exit = [int(idx) for idx in exit_order[t] if clean[int(idx)]]
        else:
            ranked_exit = ranked
        best_unheld = max(
            (
                float(exit_score[t, idx])
                for idx in ranked_exit
                if idx not in shares
            ),
            default=-np.inf,
        )

        turnover = 0.0
        trades = 0
        bought_indices_today: set[int] = set()
        mandatory_sells = []
        score_sells = []
        score_sell_required_advantages = {}
        renewal_rebalances = []
        min_hold_days = (
            int(definition["min_hold_days"])
            if min_hold_days_override is None
            else max(int(min_hold_days_override[t]), 0)
        )
        for idx in sorted(shares, key=lambda item: stocks[item]):
            age = t - entry_index[idx]
            total_age = t - original_entry_index[idx]
            absolute_max_reached = (
                absolute_max_hold_days_override is not None
                and total_age >= max(int(absolute_max_hold_days_override), 1)
            )
            current = (
                float(exit_score[t, idx])
                if np.isfinite(exit_score[t, idx])
                else -np.inf
            )
            sell_score_below = (
                float(definition["sell_score_below"])
                if sell_score_below_override is None
                else float(sell_score_below_override[t])
            )
            if sell_score_below_age_bands_override is not None:
                for min_age, value in sell_score_below_age_bands_override:
                    if age >= int(min_age):
                        sell_score_below = float(value)
            replacement_advantage = (
                float(definition["replacement_advantage"])
                if replacement_advantage_override is None
                else float(replacement_advantage_override)
                if np.ndim(replacement_advantage_override) == 0
                else float(replacement_advantage_override[t])
            )
            if replacement_advantage_age_bands_override is not None:
                for min_age, value in replacement_advantage_age_bands_override:
                    if age >= int(min_age):
                        replacement_advantage = float(value)
            current_signal_close = float(arrays["close_qfq"][t, idx])
            if np.isfinite(current_signal_close) and current_signal_close > 0:
                prior_peak = peak_signal_close.get(idx, current_signal_close)
                if not np.isfinite(prior_peak) or prior_peak <= 0:
                    prior_peak = current_signal_close
                peak_signal_close[idx] = max(prior_peak, current_signal_close)
            trailing_drawdown_exit_condition = (
                price_peak_drawdown_exit_override is not None
                and age >= min_hold_days
                and np.isfinite(current_signal_close)
                and current_signal_close > 0
                and peak_signal_close.get(idx, current_signal_close) > 0
                and 1.0
                - current_signal_close / peak_signal_close[idx]
                >= float(price_peak_drawdown_exit_override)
                and best_unheld - current >= replacement_advantage
            )
            entry_loss_exit_condition = (
                entry_price_loss_exit_override is not None
                and age >= min_hold_days
                and bool(valid_open[idx])
                and entry_raw_price.get(idx, 0.0) > 0
                and 1.0 - float(opens[idx]) / entry_raw_price[idx]
                >= float(entry_price_loss_exit_override)
                and best_unheld - current >= replacement_advantage
            )
            score_exit_condition = (
                age >= min_hold_days
                and bool(score_exit_confirmation[t, idx])
                and current < sell_score_below
                and best_unheld - current >= replacement_advantage
            ) or trailing_drawdown_exit_condition or entry_loss_exit_condition
            sell_streak[idx] = sell_streak.get(idx, 0) + 1 if score_exit_condition else 0
            score_exit = score_exit_condition and sell_streak[idx] >= max(int(sell_confirmation_days_override), 1)
            renewal_policy = str(max_hold_renewal_policy_override)
            if renewal_policy not in {
                "none",
                "top1",
                "no_score_exit",
                "top1_daily",
                "no_score_exit_daily",
                "top1_rebalance",
                "no_score_exit_rebalance",
            }:
                raise ValueError(f"未知最大持有期续持策略：{renewal_policy}")
            renew_at_max_hold = False
            if age >= max_hold and max_hold_renewal_score_override is not None:
                renewal_score_threshold = (
                    float(max_hold_renewal_score_override)
                    if np.ndim(max_hold_renewal_score_override) == 0
                    else float(max_hold_renewal_score_override[t])
                )
                renew_at_max_hold = current >= renewal_score_threshold
            elif age >= max_hold and renewal_policy in {
                "top1",
                "top1_daily",
                "top1_rebalance",
            }:
                renew_at_max_hold = bool(ranked_exit) and ranked_exit[0] == idx
            elif age >= max_hold and renewal_policy in {
                "no_score_exit",
                "no_score_exit_daily",
                "no_score_exit_rebalance",
            }:
                renew_at_max_hold = not score_exit_condition
            if renew_at_max_hold and not absolute_max_reached:
                if renewal_policy in {
                    "top1",
                    "no_score_exit",
                    "top1_rebalance",
                    "no_score_exit_rebalance",
                }:
                    entry_index[idx] = t
                    peak_signal_close[idx] = current_signal_close
                if renewal_policy in {
                    "top1_rebalance",
                    "no_score_exit_rebalance",
                }:
                    renewal_rebalances.append(idx)
                sell_streak[idx] = 0
                continue
            if absolute_max_reached or age >= max_hold:
                mandatory_sells.append(idx)
            elif score_exit:
                score_sells.append(idx)
                score_sell_required_advantages[idx] = replacement_advantage
        base_limit = 0
        if max_daily_score_sells_override is not None:
            base_limit = max(
                int(max_daily_score_sells_override)
                if np.ndim(max_daily_score_sells_override) == 0
                else int(max_daily_score_sells_override[t]),
                0,
            )
            pressure_trigger_for_day = (
                None
                if score_sell_pressure_trigger_override is None
                else int(score_sell_pressure_trigger_override)
                if np.ndim(score_sell_pressure_trigger_override) == 0
                else int(score_sell_pressure_trigger_override[t])
            )
            pressure_limit_for_day = (
                None
                if score_sell_pressure_limit_override is None
                else int(score_sell_pressure_limit_override)
                if np.ndim(score_sell_pressure_limit_override) == 0
                else int(score_sell_pressure_limit_override[t])
            )
            pressure_confirmation_days_for_day = (
                int(score_sell_pressure_confirmation_days_override)
                if np.ndim(score_sell_pressure_confirmation_days_override) == 0
                else int(score_sell_pressure_confirmation_days_override[t])
            )
            pressure_trigger_met = (
                pressure_trigger_for_day is not None
                and len(score_sells) >= pressure_trigger_for_day
            )
            score_sell_pressure_streak = (
                score_sell_pressure_streak + 1 if pressure_trigger_met else 0
            )
            limit = resolve_score_sell_limit(
                base_limit,
                len(score_sells),
                pressure_trigger_for_day,
                pressure_limit_for_day,
                pressure_streak=score_sell_pressure_streak,
                confirmation_days=pressure_confirmation_days_for_day,
            )
            limit = apply_pressure_cooldown(
                limit,
                base_limit,
                t,
                last_pressure_extra_sell_index,
                score_sell_pressure_cooldown_days_override,
            )
            qualified_replacement_count = None
            replacement_quality_threshold = None
            if score_sell_pressure_min_replacement_score_override is not None:
                replacement_quality_threshold = float(
                    score_sell_pressure_min_replacement_score_override
                )
                reentry_cooldown_days = max(int(reentry_cooldown_days_override), 0)
                qualified_replacement_count = sum(
                    1
                    for idx in entry_ranked
                    if idx not in shares
                    and t - last_sell_index.get(idx, -10**9)
                    >= reentry_cooldown_days
                    and np.isfinite(score[t, idx])
                    and float(score[t, idx]) >= replacement_quality_threshold
                )
                limit = apply_pressure_replacement_quality_gate(
                    limit,
                    base_limit,
                    qualified_replacement_count,
                )
            if score_sell_pressure_observer is not None:
                score_sell_pressure_observer(
                    {
                        "signal_date": signal_date,
                        "buy_date": buy_date,
                        "score_sell_candidate_count": int(len(score_sells)),
                        "base_limit": int(base_limit),
                        "resolved_limit": int(limit),
                        "pressure_trigger": pressure_trigger_for_day,
                        "pressure_limit": pressure_limit_for_day,
                        "pressure_streak": int(score_sell_pressure_streak),
                        "confirmation_days": pressure_confirmation_days_for_day,
                        "cooldown_days": int(
                            score_sell_pressure_cooldown_days_override
                        ),
                        "cooldown_active": bool(
                            int(score_sell_pressure_cooldown_days_override) > 0
                            and t - last_pressure_extra_sell_index
                            <= int(score_sell_pressure_cooldown_days_override)
                        ),
                        "replacement_quality_threshold": replacement_quality_threshold,
                        "qualified_replacement_count": qualified_replacement_count,
                    }
                )
            ordered_score_sells = order_score_sells(
                score_sells,
                exit_score[t],
                stocks,
                score_sell_priority[t],
            )
            pressure_extra_min_age_for_day = (
                None
                if score_sell_pressure_extra_min_age_override is None
                else int(score_sell_pressure_extra_min_age_override)
                if np.ndim(score_sell_pressure_extra_min_age_override) == 0
                else int(score_sell_pressure_extra_min_age_override[t])
            )
            if pairwise_replacement_guard_override:
                reentry_cooldown_days = max(int(reentry_cooldown_days_override), 0)
                replacement_candidates = [
                    idx
                    for idx in entry_ranked
                    if idx not in shares
                    and not bool(entry_block[t, idx])
                    and t - last_sell_index.get(idx, -10**9)
                    >= reentry_cooldown_days
                ]
                score_sells = pairwise_supported_score_sells(
                    ordered_score_sells,
                    replacement_candidates,
                    exit_score[t],
                    exit_score[t],
                score_sell_required_advantages,
                limit,
                )
            else:
                score_sells = pressure_sells_with_extra_min_age(
                    ordered_score_sells,
                    limit,
                    base_limit,
                    entry_index,
                    t,
                    pressure_extra_min_age_for_day,
                )
        selected_score_sells = set(score_sells)
        executed_score_sell_count = 0
        sells = mandatory_sells + score_sells
        for idx in sells:
            if not valid_open[idx] or opens[idx] <= pre_close[idx] * (1.0 - board_rate[idx]) * 1.005:
                continue
            if (
                defer_sell_on_limit_up_override
                and opens[idx] >= pre_close[idx] * (1.0 + board_rate[idx]) * 0.995
            ):
                continue
            gross = float(shares[idx] * opens[idx])
            cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            turnover += gross
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx])})
            del shares[idx], entry_index[idx], original_entry_index[idx]
            entry_raw_price.pop(idx, None)
            last_price.pop(idx, None)
            peak_signal_close.pop(idx, None)
            sell_streak.pop(idx, None)
            last_sell_index[idx] = t
            if idx in selected_score_sells:
                executed_score_sell_count += 1
        if executed_score_sell_count > base_limit:
            last_pressure_extra_sell_index = t

        if breadth_count_override is None:
            breadth_count = int(np.sum(np.isfinite(score[t]) & (score[t] >= float(protocol["breadth"]["score_threshold"]))))
        else:
            breadth_count = int(breadth_count_override[t])
        if gross_target_override is None:
            state = protocol["breadth_state"]
            above = breadth_count >= int(state["breadth_count_threshold"])
            high_state = above if state["high_when"] == "above" else not above
            gross_target = float(state["high_gross"] if high_state else state["low_gross"])
        else:
            gross_target = float(gross_target_override[t])
        target_cohorts = cohorts if target_cohorts_override is None else max(int(target_cohorts_override), 1)
        target_pct = (
            gross_target / (target_cohorts * top_n)
            if target_pct_override is None
            else max(float(target_pct_override[t]), 0.0)
        )
        if max_positions_schedule_override is not None:
            max_positions = max(int(max_positions_schedule_override[t]), top_n)
        elif max_positions_override is not None:
            max_positions = max(int(max_positions_override), top_n)
        else:
            max_positions = cohorts * top_n

        portfolio_rebalance_active = (
            portfolio_rebalance_active_override is not None
            and bool(portfolio_rebalance_active_override[t])
        )
        rebalance_indices = (
            list(shares) if portfolio_rebalance_active else renewal_rebalances
        )
        rebalance_plan = []
        for idx in rebalance_indices:
            if not valid_open[idx]:
                continue
            raw_open = float(opens[idx])
            rebalance_target_pct = target_pct
            if maintenance_target_multiplier_override is not None:
                rebalance_target_pct *= max(
                    float(maintenance_target_multiplier_override[t, idx]), 0.0
                )
            desired_quantity = (
                math.floor(
                    equity_before * rebalance_target_pct / raw_open / 100.0
                )
                * 100.0
                if raw_open > 0
                else 0.0
            )
            current_quantity = float(shares[idx])
            if portfolio_rebalance_active:
                current_weight = current_quantity * raw_open / max(equity_before, 1.0)
                deviation = current_weight - rebalance_target_pct
                threshold = (
                    portfolio_rebalance_overweight_deviation_override
                    if deviation >= 0.0
                    else portfolio_rebalance_underweight_deviation_override
                )
                if threshold is None:
                    threshold = portfolio_rebalance_min_weight_deviation_override
                if abs(deviation) < float(threshold):
                    continue
            rebalance_plan.append(
                (
                    idx,
                    raw_open,
                    desired_quantity,
                    current_quantity,
                    rebalance_target_pct,
                )
            )
        if portfolio_rebalance_active:
            rebalance_plan.sort(
                key=lambda item: (item[2] >= item[3], stocks[item[0]])
            )
        for (
            idx,
            raw_open,
            desired_quantity,
            current_quantity,
            rebalance_target_pct,
        ) in rebalance_plan:
            if desired_quantity < current_quantity:
                quantity = current_quantity - desired_quantity
                gross = quantity * raw_open
                cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
                shares[idx] = desired_quantity
                turnover += gross
                trades += 1
                if record_actions:
                    actions.append(
                        {
                            "signal_date": signal_date,
                            "buy_date": buy_date,
                            "action": "SELL",
                            "stock_code": stocks[idx],
                            "target_pct": rebalance_target_pct,
                            "execution_open_raw": raw_open,
                        }
                    )
            elif desired_quantity > current_quantity:
                if not maintenance_buy_allowed(idx, maintenance_block[t]):
                    continue
                requested = desired_quantity - current_quantity
                price = raw_open * (1.0 + slip)
                affordable = math.floor(cash / (price * 1.0003) / 100.0) * 100.0
                quantity = min(requested, affordable)
                spend = quantity * price * 1.0003
                if quantity >= 100 and spend <= cash + 1e-6:
                    cash -= spend
                    shares[idx] = current_quantity + quantity
                    bought_indices_today.add(idx)
                    turnover += quantity * raw_open
                    trades += 1
                    if record_actions:
                        actions.append(
                            {
                                "signal_date": signal_date,
                                "buy_date": buy_date,
                                "action": "BUY",
                                "stock_code": stocks[idx],
                                "target_pct": rebalance_target_pct,
                                "execution_open_raw": raw_open,
                            }
                        )

        available_slots = max(max_positions - len(shares), 0)
        if entry_slots_schedule_override is None:
            slots = min(top_n, available_slots)
        else:
            slots = min(
                max(int(entry_slots_schedule_override[t]), 0),
                available_slots,
            )
        if refill_after_sells_override and sells:
            refill_limit = len(sells)
            if max_refill_buys_override is not None:
                refill_limit = min(refill_limit, max(int(max_refill_buys_override), 0))
            slots = min(available_slots, max(top_n, refill_limit))
        buy_ranked = (
            ranked_exit
            if daily_replacement and sells and not entry_ranked
            else entry_ranked
        )
        buy_slots = (
            min(slots, len(sells))
            if daily_replacement and sells and not entry_ranked
            else slots
        )
        reentry_cooldown_days = max(int(reentry_cooldown_days_override), 0)
        candidates = [
            idx
            for idx in buy_ranked
            if idx not in shares
            and not bool(entry_block[t, idx])
            and t - last_sell_index.get(idx, -10**9) >= reentry_cooldown_days
        ][:buy_slots]
        if rank_weight_power > 0 and candidates:
            raw_weights = np.arange(len(candidates), 0, -1, dtype=float) ** float(rank_weight_power)
            candidate_target_pcts = gross_target / cohorts * raw_weights / raw_weights.sum()
        else:
            candidate_target_pcts = np.full(len(candidates), target_pct, dtype=float)
        if split_candidate_target_override and len(candidates) > 1:
            candidate_target_pcts /= float(len(candidates))
        if candidate_target_multiplier_override is not None and candidates:
            candidate_target_pcts *= np.asarray(
                [max(float(candidate_target_multiplier_override[t, idx]), 0.0) for idx in candidates],
                dtype=float,
            )
        planned_quantities = None
        if candidate_quantity_allocator_override is not None and candidates:
            planned_quantities = np.asarray(
                candidate_quantity_allocator_override(
                    candidates=np.asarray(candidates, dtype=np.int64),
                    target_pcts=np.asarray(candidate_target_pcts, dtype=np.float64),
                    raw_opens=np.asarray(opens[candidates], dtype=np.float64),
                    equity_before=float(equity_before),
                    cash_available=float(cash),
                    slippage_ratio=float(slip),
                ),
                dtype=np.float64,
            )
            if planned_quantities.shape != (len(candidates),):
                raise RuntimeError("candidate quantity allocator shape is invalid")
            if (
                not np.isfinite(planned_quantities).all()
                or (planned_quantities < 0.0).any()
                or not np.allclose(
                    np.mod(planned_quantities, 100.0), 0.0, rtol=0.0, atol=1e-12
                )
            ):
                raise RuntimeError("candidate quantity allocator produced invalid lots")
            planned_spend = float(
                np.sum(
                    planned_quantities
                    * np.asarray(opens[candidates], dtype=np.float64)
                    * (1.0 + slip)
                    * 1.0003
                )
            )
            if planned_spend > cash + 1e-6:
                raise RuntimeError("candidate quantity allocator exceeded available cash")
        for candidate_offset, (idx, candidate_target_pct) in enumerate(
            zip(candidates, candidate_target_pcts)
        ):
            price = float(opens[idx]) * (1.0 + slip)
            if planned_quantities is None:
                budget = min(equity_before * float(candidate_target_pct), cash / 1.001)
                quantity = math.floor(budget / (price * 1.0003) / 100.0) * 100.0
            else:
                quantity = float(planned_quantities[candidate_offset])
            spend = quantity * price * 1.0003
            if quantity < 100 or spend > cash + 1e-6:
                continue
            cash -= spend
            shares[idx], entry_index[idx], original_entry_index[idx] = quantity, t, t
            bought_indices_today.add(idx)
            entry_raw_price[idx] = float(opens[idx])
            last_price[idx] = float(opens[idx])
            peak_signal_close[idx] = np.nan
            sell_streak[idx] = 0
            turnover += quantity * float(opens[idx])
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": float(candidate_target_pct), "execution_open_raw": float(opens[idx])})

        if (
            residual_cash_sweep_override
            and shares
            and residual_cash_sweep_triggered(
                residual_cash_sweep_trigger_override, trades, bought_indices_today
            )
        ):
            mark_prices = np.asarray(
                [float(last_price.get(idx, np.nan)) for idx in range(len(stocks))],
                dtype=np.float64,
            )
            eligible_indices = {
                idx
                for idx in shares
                if valid_open[idx]
                and maintenance_buy_allowed(idx, maintenance_block[t])
            }
            residual_target_pct = gross_target / max(max_positions, 1)
            cash, additions = residual_cash_sweep_plan(
                stocks=stocks,
                shares=shares,
                raw_opens=opens,
                mark_prices=mark_prices,
                eligible_indices=eligible_indices,
                cash=cash,
                target_pct=residual_target_pct,
                slippage_ratio=slip,
                max_additions=residual_cash_sweep_max_orders_override,
                recipient_mode=residual_cash_sweep_recipient_override,
                priority_scores=score[t],
            )
            for idx, quantity in additions.items():
                shares[idx] += quantity
                turnover += quantity * float(opens[idx])
                if idx not in bought_indices_today:
                    trades += 1
                if record_actions and idx not in bought_indices_today:
                    actions.append(
                        {
                            "signal_date": signal_date,
                            "buy_date": buy_date,
                            "action": "BUY",
                            "stock_code": stocks[idx],
                            "target_pct": float(residual_target_pct),
                            "execution_open_raw": float(opens[idx]),
                        }
                    )
                bought_indices_today.add(idx)

        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        if position_observer is not None:
            position_observer(
                {
                    "signal_date": signal_date,
                    "buy_date": buy_date,
                    "previous_equity": float(previous_equity),
                    "equity_before_trades": float(equity_before),
                    "equity_after_trades": float(equity_after),
                    "cash_after_trades": float(cash),
                    "mark_records": mark_records,
                    "positions_after_trades": {
                        stocks[idx]: {
                            "shares": float(shares[idx]),
                            "mark_price": float(last_price.get(idx, 0.0)),
                        }
                        for idx in sorted(shares, key=lambda item: stocks[item])
                    },
                }
            )
        previous_equity = equity_after

    daily = pd.DataFrame(rows)
    return (daily, pd.DataFrame(actions, columns=ACTION_COLUMNS)) if record_actions else daily


def main() -> None:
    parser = argparse.ArgumentParser(description="分数宽度策略独立卖出规则研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    cfg = protocol["score"]
    score, order = v95.score_pair(full, cfg["weight_5d"], cfg["smooth_window"], cfg["current_weight"])

    if args.open_known_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for item in frozen["candidates"]:
            for start in protocol["known_stress_buy_starts"]:
                daily = simulate(full, score, order, item["definition"], protocol, protocol["known_stress_end"], start)
                rows.append({"case_id": item["case_id"], "buy_start": start, **core.metrics(daily, start, protocol["known_stress_end"])})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({"rows": len(frame), "candidates": frame["case_id"].nunique()}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    score_obs, order_obs = score[: len(observation["dates"])], order[: len(observation["dates"])]
    rows, definitions = [], {}
    grid = protocol["grid"]
    for min_hold, sell_below, advantage in product(grid["min_hold_days"], grid["sell_score_below"], grid["replacement_advantage"]):
        definition = {"min_hold_days": min_hold, "sell_score_below": sell_below, "replacement_advantage": advantage}
        case_id = stable_id(definition)
        definitions[case_id] = definition
        daily = simulate(observation, score_obs, order_obs, definition, protocol, protocol["observation_end"])
        rows.append({"case_id": case_id, **definition, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    gate = protocol["observation_gate"]
    frame["eligible"] = (
        (frame["min_year_cumulative_return"] >= gate["min_year_return"])
        & (frame["full_linear_annual_proxy"] >= gate["full_linear_annual_proxy"])
        & (frame["full_sharpe"] >= gate["full_sharpe"])
        & (frame["full_max_drawdown"] <= gate["max_drawdown"])
    )
    frame = frame.sort_values(["eligible", "full_linear_annual_proxy", "min_year_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, False, True])
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_this_grid_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
