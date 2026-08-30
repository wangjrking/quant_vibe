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
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_portfolio_refine_v96_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026_development.json"


def stable_id(p: dict) -> str:
    return "v96_" + hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-known-2026", action="store_true")
    opt = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    s = protocol["score"]
    score_id = "fixed_10d_smooth5_c0p5"
    score, order = v95.score_pair(full, s["weight_5d"], s["smooth_window"], s["current_weight"])
    rank_map = {score_id: (score, order)}
    if opt.open_known_2026:
        candidates = json.loads(FROZEN.read_text(encoding="utf-8"))["candidates"]
        rows = []
        for item in candidates:
            p = item["definition"]
            for start in protocol["known_2026_development_starts"]:
                daily = v86.evaluate(full, rank_map, p, {**protocol, "validation_end": protocol["known_2026_development_end"]}, start)
                rows.append({"case_id": item["case_id"], "buy_start": start, **core.metrics(daily, start, protocol["known_2026_development_end"])})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_development.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return
    observation = robust.truncate_observation(full, protocol["observation_end"])
    rank_obs = {score_id: (score[: len(observation["dates"])], order[: len(observation["dates"])])}
    fixed, grid = protocol["fixed_universe"], protocol["grid"]
    rows, definitions = [], {}
    for top_n, hold, every, gross in product(grid["top_n_per_rebalance"], grid["hold_days"], grid["rebalance_every"], grid["target_gross_exposure"]):
        p = {**fixed, "blend": score_id, "top_n_per_rebalance": top_n, "hold_days": hold, "rebalance_every": every, "target_gross_exposure": gross}
        cid = stable_id(p)
        definitions[cid] = p
        daily = v86.evaluate(observation, rank_obs, p, protocol)
        rows.append({"case_id": cid, **p, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    g = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= g["min_year_return"]) & (frame["full_linear_annual_proxy"] >= g["full_linear_annual_proxy"]) & (frame["full_sharpe"] >= g["full_sharpe"]) & (frame["full_max_drawdown"] <= g["max_drawdown"])
    frame = frame.sort_values(["eligible", "min_year_cumulative_return", "full_linear_annual_proxy", "full_sharpe"], ascending=False)
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_development", "known_2026_informed_score_choice": True, "true_unseen_forward_start": protocol["true_unseen_forward_start"], "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
