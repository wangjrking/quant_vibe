# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_breadth_exit_v109_20260722 as v109
import research_active_l4_robust_objective_v56_20260721 as robust
import research_active_l4_slow_portfolio_v120_20260722 as v120
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_slow_portfolio_refine_v121_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v121_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="正式L4慢分数单批组合窄邻域研究")
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
            current = v120.case_protocol(protocol, item["definition"])
            for start in protocol["known_stress_buy_starts"]:
                daily = v109.simulate(full, score, order, item["definition"], current, protocol["known_stress_end"], start)
                rows.append({"case_id": item["case_id"], "buy_start": start, **core.metrics(daily, start, protocol["known_stress_end"])})
        pd.DataFrame(rows).to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({"rows": len(rows), "candidates": len(frozen["candidates"])}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    score_obs, order_obs = score[: len(observation["dates"])], order[: len(observation["dates"])]
    rows, definitions = [], {}
    grid = protocol["grid"]
    for top_n, max_hold, every in product(grid["top_n_per_rebalance"], grid["max_hold_days"], grid["rebalance_every"]):
        if max_hold < every or max_hold >= every * 2:
            continue
        definition = {**protocol["fixed_exit"], "top_n_per_rebalance": top_n, "max_hold_days": max_hold, "rebalance_every": every}
        case_id = stable_id(definition)
        definitions[case_id] = definition
        daily = v109.simulate(observation, score_obs, order_obs, definition, v120.case_protocol(protocol, definition), protocol["observation_end"])
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
