# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_breadth_exit_v109_20260722 as v109
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_market_regime_v134_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v134_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def case_protocol(protocol: dict) -> dict:
    result = copy.deepcopy(protocol)
    result["fixed_strategy"] = {"top_n_per_rebalance": 2, "max_hold_days": 15, "rebalance_every": 1}
    return result


def market_momentum(arrays: dict, lookback: int) -> np.ndarray:
    close = arrays["close_qfq"].astype(float)
    result = np.full(len(close), np.nan, dtype=float)
    for t in range(lookback, len(close)):
        valid = np.isfinite(close[t]) & np.isfinite(close[t - lookback]) & (close[t - lookback] > 0)
        if valid.any():
            result[t] = float(np.nanmedian(close[t, valid] / close[t - lookback, valid] - 1.0))
    return result


def gross_schedule(arrays: dict, definition: dict) -> np.ndarray:
    momentum = market_momentum(arrays, int(definition["market_lookback"]))
    high = np.isfinite(momentum) & (momentum >= float(definition["market_return_min"]))
    return np.where(high, 1.0, float(definition["low_gross"])).astype(float)


def run_case(arrays, score, order, definition, protocol, end, start=None):
    return v109.simulate(
        arrays,
        score,
        order,
        definition,
        case_protocol(protocol),
        end,
        start,
        gross_target_override=gross_schedule(arrays, definition),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="正式L4每日重叠策略的历史市场状态门控研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(full, 0.0, int(protocol["score"]["smooth_window"]), float(protocol["score"]["current_weight"]))

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
    grid = protocol["grid"]
    momentum_cache = {lookback: market_momentum(observation, int(lookback)) for lookback in grid["market_lookback"]}
    for lookback, return_min, low_gross in product(grid["market_lookback"], grid["market_return_min"], grid["low_gross"]):
        definition = {**protocol["fixed_definition"], "market_lookback": lookback, "market_return_min": return_min, "low_gross": low_gross}
        case_id = stable_id(definition)
        definitions[case_id] = definition
        momentum = momentum_cache[lookback]
        high_days = int(np.sum(np.isfinite(momentum) & (momentum >= float(return_min))))
        daily = run_case(observation, score_obs, order_obs, definition, protocol, protocol["observation_end"])
        rows.append({"case_id": case_id, **definition, "high_days": high_days, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    gate = protocol["observation_gate"]
    frame["eligible"] = (
        (frame["min_year_cumulative_return"] >= gate["min_year_return"])
        & (frame["full_linear_annual_proxy"] >= gate["full_linear_annual_proxy"])
        & (frame["full_sharpe"] >= gate["full_sharpe"])
        & (frame["full_max_drawdown"] <= gate["max_drawdown"])
        & (frame["high_days"] >= gate["min_high_days"])
    )
    frame = frame.sort_values(["eligible", "full_linear_annual_proxy", "min_year_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, False, True])
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_this_grid_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
