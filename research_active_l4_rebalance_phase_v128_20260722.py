# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_breadth_exit_v109_20260722 as v109
import research_active_l4_robust_objective_v56_20260721 as robust
import research_active_l4_slow_portfolio_breadth_v122_20260722 as v122
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_rebalance_phase_v128_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(full, 0.0, int(protocol["score"]["smooth_window"]), float(protocol["score"]["current_weight"]))
    definition = protocol["definition"]
    current = v122.case_protocol(protocol, definition)
    observation = robust.truncate_observation(full, protocol["observation_end"])
    score_obs, order_obs = score[: len(observation["dates"])], order[: len(observation["dates"])]
    rows = []
    for offset in range(int(protocol["fixed_strategy"]["rebalance_every"])):
        daily_obs = v109.simulate(observation, score_obs, order_obs, definition, current, protocol["observation_end"], rebalance_offset=offset)
        obs_metrics = robust.evaluate_robust(daily_obs, protocol)
        for start in protocol["known_stress_buy_starts"]:
            daily = v109.simulate(full, score, order, definition, current, protocol["known_stress_end"], start, rebalance_offset=offset)
            rows.append({"rebalance_offset": offset, "window": "known_2026", "buy_start": start, **core.metrics(daily, start, protocol["known_stress_end"]), "observation_linear_annual_proxy": obs_metrics["full_linear_annual_proxy"], "observation_sharpe": obs_metrics["full_sharpe"], "observation_max_drawdown": obs_metrics["full_max_drawdown"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "phase_results.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby("rebalance_offset", as_index=False).agg(
        observation_linear_annual_proxy=("observation_linear_annual_proxy", "first"),
        observation_sharpe=("observation_sharpe", "first"),
        observation_max_drawdown=("observation_max_drawdown", "first"),
        known_2026_min_annual=("linear_annual_proxy", "min"),
        known_2026_median_annual=("linear_annual_proxy", "median"),
        known_2026_min_sharpe=("sharpe", "min"),
        known_2026_max_drawdown=("max_drawdown", "max"),
    )
    summary.to_csv(OUT / "phase_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    main()
