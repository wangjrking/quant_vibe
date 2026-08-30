# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_cost_aware_v98_20260722 as v98
import research_active_l4_rank_smoothing_v94_20260722 as v94
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_score_cost_v100_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v100_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def score_pair(arrays: dict, definition: dict) -> tuple[np.ndarray, np.ndarray]:
    w3 = float(definition["weight_3d"])
    w5 = float(definition["weight_5d"])
    raw = (w3 * arrays["rank_3d"] + w5 * arrays["rank_5d"] + (1.0 - w3 - w5) * arrays["rank_10d"]).astype(np.float32)
    smooth = v94.rolling_nanmean(raw, int(definition["smooth_window"]))
    current = float(definition["current_weight"])
    score = (current * raw + (1.0 - current) * smooth).astype(np.float32)
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")
    return score, order


def main() -> None:
    parser = argparse.ArgumentParser(description="固定低换手组合下的多周期评分研究")
    parser.add_argument("--open-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    fixed = protocol["fixed_strategy"]
    if args.open_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for item in frozen["candidates"]:
            score, order = score_pair(full, item["score_definition"])
            for start in protocol["validation_buy_starts"]:
                daily = v98.evaluate(full, score, order, fixed, protocol, start)
                rows.append({"case_id": item["case_id"], "buy_start": start, **core.metrics(daily, start, protocol["validation_end"])})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    grid = protocol["score_grid"]
    rows, definitions = [], {}
    for w3, w5, window, current in product(
        grid["weight_3d"], grid["weight_5d"], grid["smooth_window"], grid["current_weight"]
    ):
        if w3 + w5 > float(grid["max_short_weight"]):
            continue
        definition = {"weight_3d": w3, "weight_5d": w5, "smooth_window": window, "current_weight": current}
        case_id = stable_id(definition)
        definitions[case_id] = definition
        score, order = score_pair(observation, definition)
        daily = v98.evaluate(observation, score, order, fixed, protocol)
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
        {"case_id": str(row.case_id), "score_definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()}
        for _, row in selected.iterrows()
    ]
    FROZEN.write_text(
        json.dumps({"status": "frozen_before_2026", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
