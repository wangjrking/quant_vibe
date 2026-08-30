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
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_dual_speed_v116_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v116_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_scores(arrays: dict, definition: dict, protocol: dict):
    fast = protocol["fast_score"]
    slow = protocol["slow_score"]
    fast_score, _ = v95.score_pair(arrays, 0.0, int(fast["smooth_window"]), float(fast["current_weight"]))
    slow_score, _ = v95.score_pair(arrays, 0.0, int(slow["smooth_window"]), float(definition["slow_current_weight"]))
    breadth_count = np.sum(np.isfinite(fast_score) & (fast_score >= float(protocol["breadth_score_threshold"])), axis=1).astype(np.int32)
    high = breadth_count >= int(definition["breadth_count_threshold"])
    selected = np.where(high[:, None], fast_score, slow_score).astype(np.float32)
    order = np.argsort(-np.nan_to_num(selected, nan=-np.inf), axis=1).astype(np.int32)
    return selected, order, breadth_count


def case_protocol(protocol: dict, definition: dict) -> dict:
    result = copy.deepcopy(protocol)
    result["breadth"] = {"score_threshold": float(protocol["breadth_score_threshold"])}
    result["breadth_state"] = {
        **protocol["fixed_breadth"],
        "breadth_count_threshold": int(definition["breadth_count_threshold"]),
        "low_gross": float(definition["low_gross"]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="正常宽度快评分与极窄宽度慢评分切换研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}

    if args.open_known_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for item in frozen["candidates"]:
            score, order, breadth_count = build_scores(full, item["definition"], protocol)
            current = case_protocol(protocol, item["definition"])
            for start in protocol["known_stress_buy_starts"]:
                daily = v109.simulate(full, score, order, item["definition"], current, protocol["known_stress_end"], start, breadth_count_override=breadth_count)
                rows.append({"case_id": item["case_id"], "buy_start": start, **core.metrics(daily, start, protocol["known_stress_end"])})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({"rows": len(frame), "candidates": frame["case_id"].nunique()}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    rows, definitions = [], {}
    grid = protocol["grid"]
    for count_threshold, slow_weight, low_gross in product(grid["breadth_count_threshold"], grid["slow_current_weight"], grid["low_gross"]):
        definition = {**protocol["fixed_exit"], "breadth_count_threshold": count_threshold, "slow_current_weight": slow_weight, "low_gross": low_gross}
        case_id = stable_id(definition)
        definitions[case_id] = definition
        score, order, breadth_count = build_scores(observation, definition, protocol)
        daily = v109.simulate(observation, score, order, definition, case_protocol(protocol, definition), protocol["observation_end"], breadth_count_override=breadth_count)
        rows.append({"case_id": case_id, **definition, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    gate = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= gate["min_year_return"]) & (frame["full_linear_annual_proxy"] >= gate["full_linear_annual_proxy"]) & (frame["full_sharpe"] >= gate["full_sharpe"]) & (frame["full_max_drawdown"] <= gate["max_drawdown"])
    frame = frame.sort_values(["eligible", "full_linear_annual_proxy", "min_year_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, False, True])
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_this_grid_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
