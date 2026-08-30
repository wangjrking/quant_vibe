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
import research_active_l4_relaxed_universe_v86_20260722 as v86
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_concentration_v97_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v97_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="10D 平滑评分集中度预注册研究")
    parser.add_argument("--open-2026", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    score_cfg = protocol["score"]
    score_id = "fixed_10d_smooth5_c0p5"
    score, order = v95.score_pair(
        full,
        score_cfg["weight_5d"],
        score_cfg["smooth_window"],
        score_cfg["current_weight"],
    )
    ranks = {score_id: (score, order)}

    if args.open_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for item in frozen["candidates"]:
            for start in protocol["validation_buy_starts"]:
                daily = v86.evaluate(full, ranks, item["definition"], protocol, start)
                rows.append(
                    {
                        "case_id": item["case_id"],
                        "buy_start": start,
                        **core.metrics(daily, start, protocol["validation_end"]),
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    ranks_obs = {score_id: (score[: len(observation["dates"])], order[: len(observation["dates"])])}
    fixed = protocol["fixed_universe"]
    grid = protocol["grid"]
    rows = []
    definitions = {}
    for top_n, hold, every, gross in product(
        grid["top_n_per_rebalance"],
        grid["hold_days"],
        grid["rebalance_every"],
        grid["target_gross_exposure"],
    ):
        definition = {
            **fixed,
            "blend": score_id,
            "top_n_per_rebalance": top_n,
            "hold_days": hold,
            "rebalance_every": every,
            "target_gross_exposure": gross,
        }
        case_id = stable_id(definition)
        definitions[case_id] = definition
        daily = v86.evaluate(observation, ranks_obs, definition, protocol)
        rows.append({"case_id": case_id, **definition, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    gate = protocol["observation_gate"]
    frame["eligible"] = (
        (frame["min_year_cumulative_return"] >= gate["min_year_return"])
        & (frame["full_linear_annual_proxy"] >= gate["full_linear_annual_proxy"])
        & (frame["full_sharpe"] >= gate["full_sharpe"])
        & (frame["full_max_drawdown"] <= gate["max_drawdown"])
    )
    frame = frame.sort_values(
        ["eligible", "full_linear_annual_proxy", "min_year_cumulative_return", "full_sharpe", "case_id"],
        ascending=[False, False, False, False, True],
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
        json.dumps(
            {
                "status": "frozen_before_2026",
                "known_2026_used_for_selection": False,
                "true_unseen_forward_start": protocol["true_unseen_forward_start"],
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
