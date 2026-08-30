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
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_phase_ensemble_v130_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"


def ensemble(arrays, score, order, definition, protocol, end, start=None):
    frames = []
    current = v122.case_protocol(protocol, definition)
    for offset in range(int(protocol["fixed_strategy"]["rebalance_every"])):
        daily = v109.simulate(arrays, score, order, definition, current, end, start, rebalance_offset=offset)
        frames.append(daily.set_index("date")["return"].rename(str(offset)))
    returns = pd.concat(frames, axis=1).fillna(0.0).mean(axis=1)
    equity = float(protocol["execution"]["initial_cash"]) * (1.0 + returns).cumprod()
    return pd.DataFrame({"date": returns.index.astype(str), "return": returns.to_numpy(), "equity": equity.to_numpy(), "turnover": 0.0, "invested_ratio": 0.0, "positions": 0, "trades": 0})


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(full, 0.0, int(protocol["score"]["smooth_window"]), float(protocol["score"]["current_weight"]))
    definition = protocol["definition"]
    observation = robust.truncate_observation(full, protocol["observation_end"])
    daily_obs = ensemble(observation, score[: len(observation["dates"])], order[: len(observation["dates"])], definition, protocol, protocol["observation_end"])
    observation_metrics = robust.evaluate_robust(daily_obs, protocol)
    stress = []
    for start in protocol["known_stress_buy_starts"]:
        daily = ensemble(full, score, order, definition, protocol, protocol["known_stress_end"], start)
        stress.append({"buy_start": start, **core.metrics(daily, start, protocol["known_stress_end"])})
    pd.DataFrame(stress).to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
    result = {"status": "research_only_all_phase_equal_weight", "observation": observation_metrics, "known_2026": stress, "production_change_allowed": False}
    (OUT / "ensemble_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
