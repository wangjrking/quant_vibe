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
import research_active_l4_adaptive_exit_v162_20260722 as v162
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_regime_horizon_v172_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v172_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Active L4 market-regime horizon study")
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
                daily = v162.run_case(
                    full, score, order, item["definition"], protocol,
                    protocol["known_stress_end"], start,
                )
                rows.append({
                    "case_id": item["case_id"],
                    "buy_start": start,
                    **core.metrics(daily, start, protocol["known_stress_end"]),
                })
        pd.DataFrame(rows).to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({"rows": len(rows), "candidates": len(frozen["candidates"])}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    score_obs = score[: len(observation["dates"])]
    order_obs = order[: len(observation["dates"])]
    rows, definitions = [], {}
    for lookback, daily_threshold in product(
        protocol["grid"]["market_lookback"],
        protocol["grid"]["strong_daily_return_threshold"],
    ):
        definition = {
            **protocol["fixed_definition"],
            "market_lookback": int(lookback),
            "strong_return_min": float(lookback) * float(daily_threshold),
            "strong_daily_return_threshold": float(daily_threshold),
        }
        case_id = stable_id(definition)
        definitions[case_id] = definition
        daily = v162.run_case(
            observation, score_obs, order_obs, definition, protocol,
            protocol["observation_end"],
        )
        rows.append({"case_id": case_id, **definition, **robust.evaluate_robust(daily, protocol)})

    frame = pd.DataFrame(rows)
    gate = protocol["observation_gate"]
    frame["eligible"] = (
        (frame["min_year_cumulative_return"] >= gate["min_year_return"])
        & (frame["full_linear_annual_proxy"] >= gate["full_linear_annual_proxy"])
        & (frame["full_sharpe"] >= gate["full_sharpe"])
        & (frame["full_max_drawdown"] <= gate["max_drawdown"])
        & (frame["recent60_linear_annual_proxy"] >= gate["recent60_linear_annual_proxy"])
        & (frame["recent120_linear_annual_proxy"] >= gate["recent120_linear_annual_proxy"])
    )
    frame = frame.sort_values(
        ["eligible", "full_linear_annual_proxy", "min_year_cumulative_return", "full_sharpe"],
        ascending=[False, False, False, False],
    )
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [
        {
            "case_id": str(row.case_id),
            "definition": definitions[str(row.case_id)],
            "observation_metrics": row.to_dict(),
        }
        for _, row in selected.iterrows()
    ]
    FROZEN.write_text(
        json.dumps({
            "status": "frozen_before_known_2026_stress",
            "known_2026_used_for_this_grid_selection": False,
            "candidates": candidates,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
