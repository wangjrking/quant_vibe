# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_breadth_exit_v109_20260722 as v109
import research_active_l4_market_regime_v134_20260722 as v134
import research_active_l4_robust_objective_v56_20260721 as robust
import research_active_l4_strong_trend_concentration_v153_20260722 as v153
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_adaptive_exit_v162_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v162_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def min_hold_schedule(arrays: dict, policy: str) -> np.ndarray:
    momentum = v134.market_momentum(arrays, 10)
    result = np.full(len(momentum), 4, dtype=np.int16)
    strong = np.isfinite(momentum) & (momentum >= 0.04)
    normal = np.isfinite(momentum) & (momentum >= 0.0)
    weak = np.isfinite(momentum) & (momentum < 0.0)
    if policy == "strong_3_else_4":
        result[strong] = 3
    elif policy == "normal_3_weak_4":
        result[normal] = 3
    elif policy == "weak_3_else_4":
        result[weak] = 3
    elif policy == "strong_3_normal_4_weak_5":
        result[strong] = 3
        result[weak] = 5
    elif policy != "fixed_4":
        raise ValueError("未知最短持有期策略：{}".format(policy))
    return result


def case_protocol(protocol: dict, definition: dict) -> dict:
    result = copy.deepcopy(protocol)
    result["fixed_strategy"] = {
        "top_n_per_rebalance": 1,
        "max_hold_days": int(definition["max_hold_days"]),
        "rebalance_every": 1,
    }
    return result


def run_case(
    arrays,
    score,
    order,
    definition,
    protocol,
    end,
    start=None,
    record_actions=False,
    selection_mask_override=None,
    candidate_target_multiplier_override=None,
    sell_confirmation_days_override=1,
    sell_score_below_override=None,
    replacement_advantage_override=None,
    replacement_advantage_age_bands_override=None,
    min_hold_days_schedule_override=None,
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
):
    gross, target = v153.schedules(arrays, definition)
    return v109.simulate(
        arrays,
        score,
        order,
        definition,
        case_protocol(protocol, definition),
        end,
        start,
        gross_target_override=gross,
        target_cohorts_override=4,
        target_pct_override=target,
        min_hold_days_override=(
            min_hold_schedule(arrays, definition["min_hold_policy"])
            if min_hold_days_schedule_override is None
            else min_hold_days_schedule_override
        ),
        record_actions=record_actions,
        selection_mask_override=selection_mask_override,
        candidate_target_multiplier_override=candidate_target_multiplier_override,
        sell_confirmation_days_override=sell_confirmation_days_override,
        sell_score_below_override=sell_score_below_override,
        replacement_advantage_override=replacement_advantage_override,
        replacement_advantage_age_bands_override=replacement_advantage_age_bands_override,
        max_positions_override=max_positions_override,
        max_positions_schedule_override=max_positions_schedule_override,
        entry_slots_schedule_override=entry_slots_schedule_override,
        max_daily_score_sells_override=max_daily_score_sells_override,
        refill_after_sells_override=refill_after_sells_override,
        max_refill_buys_override=max_refill_buys_override,
        split_candidate_target_override=split_candidate_target_override,
        reentry_cooldown_days_override=reentry_cooldown_days_override,
        max_hold_renewal_policy_override=max_hold_renewal_policy_override,
        min_entry_top_gap_override=min_entry_top_gap_override,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="正式L4市场状态自适应卖出研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(full, 0.0, 7, 0.1)

    if args.open_known_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for item in frozen["candidates"]:
            for start in protocol["known_stress_buy_starts"]:
                daily = run_case(full, score, order, item["definition"], protocol, protocol["known_stress_end"], start)
                rows.append({"case_id": item["case_id"], "buy_start": start, **core.metrics(daily, start, protocol["known_stress_end"])})
        pd.DataFrame(rows).to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({"rows": len(rows), "candidates": len(frozen["candidates"])}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    score_obs, order_obs = score[: len(observation["dates"])], order[: len(observation["dates"])]
    rows, definitions = [], {}
    for policy in protocol["grid"]["min_hold_policy"]:
        definition = {**protocol["fixed_definition"], "min_hold_policy": policy}
        case_id = stable_id(definition)
        definitions[case_id] = definition
        daily = run_case(observation, score_obs, order_obs, definition, protocol, protocol["observation_end"])
        rows.append({"case_id": case_id, **definition, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    frame["eligible"] = (
        (frame["min_year_cumulative_return"] >= 0.0)
        & (frame["full_linear_annual_proxy"] >= 0.08)
        & (frame["full_sharpe"] >= 0.5)
        & (frame["full_max_drawdown"] <= 0.45)
        & (frame["recent60_linear_annual_proxy"] >= 0.0)
        & (frame["recent120_linear_annual_proxy"] >= 0.1)
    )
    frame = frame.sort_values(["eligible", "full_linear_annual_proxy", "min_year_cumulative_return", "full_sharpe"], ascending=[False, False, False, False])
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]]
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_this_grid_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
