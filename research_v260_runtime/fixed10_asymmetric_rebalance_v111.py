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


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v109_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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
    sell_confirmation_days_override=1,
    sell_score_below_override=None,
    replacement_advantage_override=None,
    replacement_advantage_age_bands_override=None,
    max_positions_override=None,
    max_positions_schedule_override=None,
    entry_slots_schedule_override=None,
    max_daily_score_sells_override=None,
    refill_after_sells_override=False,
    max_refill_buys_override=None,
    split_candidate_target_override=False,
    reentry_cooldown_days_override=0,
    max_hold_renewal_policy_override="none",
    min_entry_top_gap_override=0.0,
    max_hold_renewal_score_override=None,
    portfolio_rebalance_active_override=None,
    portfolio_rebalance_min_weight_deviation_override=0.0,
    portfolio_rebalance_overweight_deviation_override=None,
    portfolio_rebalance_underweight_deviation_override=None,
    entry_block_mask_override=None,
    maintenance_buy_block_mask_override=None,
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
    shares, entry_index, last_price, sell_streak, last_sell_index = {}, {}, {}, {}, {}
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
    if entry_block.shape != event_shape or maintenance_block.shape != event_shape:
        raise ValueError("risk-event block mask shape mismatch")

    for t, signal_date in enumerate(dates[:-1]):
        if signal_date > end_date:
            break
        buy_date = str(dates[t + 1])
        if start and buy_date < start:
            continue
        opens, pre_close = prepared["buy_open"][t], prepared["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        for idx in list(shares):
            if valid_open[idx]:
                last_price[idx] = float(opens[idx])
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
            ranked_exit = [int(idx) for idx in order[t] if exit_clean[int(idx)]]
        else:
            ranked_exit = ranked
        best_unheld = max((float(score[t, idx]) for idx in ranked_exit if idx not in shares), default=-np.inf)

        turnover = 0.0
        trades = 0
        mandatory_sells = []
        score_sells = []
        renewal_rebalances = []
        min_hold_days = (
            int(definition["min_hold_days"])
            if min_hold_days_override is None
            else max(int(min_hold_days_override[t]), 0)
        )
        for idx in sorted(shares, key=lambda item: stocks[item]):
            age = t - entry_index[idx]
            current = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
            sell_score_below = (
                float(definition["sell_score_below"])
                if sell_score_below_override is None
                else float(sell_score_below_override[t])
            )
            replacement_advantage = (
                float(definition["replacement_advantage"])
                if replacement_advantage_override is None
                else float(replacement_advantage_override[t])
            )
            if replacement_advantage_age_bands_override is not None:
                for min_age, value in replacement_advantage_age_bands_override:
                    if age >= int(min_age):
                        replacement_advantage = float(value)
            score_exit_condition = (
                age >= min_hold_days
                and current < sell_score_below
                and best_unheld - current >= replacement_advantage
            )
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
            if renew_at_max_hold:
                if renewal_policy in {
                    "top1",
                    "no_score_exit",
                    "top1_rebalance",
                    "no_score_exit_rebalance",
                }:
                    entry_index[idx] = t
                if renewal_policy in {
                    "top1_rebalance",
                    "no_score_exit_rebalance",
                }:
                    renewal_rebalances.append(idx)
                sell_streak[idx] = 0
                continue
            if age >= max_hold:
                mandatory_sells.append(idx)
            elif score_exit:
                score_sells.append(idx)
        if max_daily_score_sells_override is not None:
            limit = max(
                int(max_daily_score_sells_override)
                if np.ndim(max_daily_score_sells_override) == 0
                else int(max_daily_score_sells_override[t]),
                0,
            )
            score_sells = sorted(
                score_sells,
                key=lambda item: (
                    float(score[t, item]) if np.isfinite(score[t, item]) else -np.inf,
                    stocks[item],
                ),
            )[:limit]
        sells = mandatory_sells + score_sells
        for idx in sells:
            if not valid_open[idx] or opens[idx] <= pre_close[idx] * (1.0 - board_rate[idx]) * 1.005:
                continue
            gross = float(shares[idx] * opens[idx])
            cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            turnover += gross
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx])})
            del shares[idx], entry_index[idx]
            last_price.pop(idx, None)
            sell_streak.pop(idx, None)
            last_sell_index[idx] = t

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
            desired_quantity = (
                math.floor(equity_before * target_pct / raw_open / 100.0) * 100.0
                if raw_open > 0
                else 0.0
            )
            current_quantity = float(shares[idx])
            if portfolio_rebalance_active:
                current_weight = current_quantity * raw_open / max(equity_before, 1.0)
                deviation = current_weight - target_pct
                threshold = (
                    portfolio_rebalance_overweight_deviation_override
                    if deviation >= 0
                    else portfolio_rebalance_underweight_deviation_override
                )
                if threshold is None:
                    threshold = portfolio_rebalance_min_weight_deviation_override
                if abs(deviation) < float(threshold):
                    continue
            rebalance_plan.append(
                (idx, raw_open, desired_quantity, current_quantity)
            )
        if portfolio_rebalance_active:
            rebalance_plan.sort(
                key=lambda item: (item[2] >= item[3], stocks[item[0]])
            )
        for idx, raw_open, desired_quantity, current_quantity in rebalance_plan:
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
                            "target_pct": target_pct,
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
                    turnover += quantity * raw_open
                    trades += 1
                    if record_actions:
                        actions.append(
                            {
                                "signal_date": signal_date,
                                "buy_date": buy_date,
                                "action": "BUY",
                                "stock_code": stocks[idx],
                                "target_pct": target_pct,
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
        for idx, candidate_target_pct in zip(candidates, candidate_target_pcts):
            budget = min(equity_before * float(candidate_target_pct), cash / 1.001)
            price = float(opens[idx]) * (1.0 + slip)
            quantity = math.floor(budget / (price * 1.0003) / 100.0) * 100.0
            spend = quantity * price * 1.0003
            if quantity < 100 or spend > cash + 1e-6:
                continue
            cash -= spend
            shares[idx], entry_index[idx], last_price[idx] = quantity, t, float(opens[idx])
            sell_streak[idx] = 0
            turnover += quantity * float(opens[idx])
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": float(candidate_target_pct), "execution_open_raw": float(opens[idx])})

        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
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
