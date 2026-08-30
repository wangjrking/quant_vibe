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
import research_active_l4_market_regime_v134_20260722 as v134
import research_active_l4_rank_smoothing_v94_20260722 as v94
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_regime_deterioration_v194_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v194_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def selection_mask(arrays: dict, definition: dict) -> np.ndarray:
    raw = arrays["rank_10d"].astype(np.float32)
    smooth = v94.rolling_nanmean(raw, 7)
    momentum = v134.market_momentum(arrays, 10)
    threshold = np.full(len(momentum), float(definition["normal_deterioration"]), dtype=float)
    threshold[np.isfinite(momentum) & (momentum < 0.0)] = float(definition["weak_deterioration"])
    threshold[np.isfinite(momentum) & (momentum >= 0.04)] = float(definition["strong_deterioration"])
    return np.isfinite(raw) & np.isfinite(smooth) & ((raw - smooth) >= -threshold[:, None])


def run_case(arrays, score, order, definition, protocol, end, start=None):
    return v162.run_case(arrays, score, order, definition, protocol, end, start, selection_mask_override=selection_mask(arrays, definition))


def main() -> None:
    parser = argparse.ArgumentParser(description="Active L4 regime deterioration study")
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
    for weak_threshold, strong_threshold in product(protocol["grid"]["weak_deterioration"], protocol["grid"]["strong_deterioration"]):
        definition = {**protocol["fixed_definition"], "weak_deterioration": float(weak_threshold), "strong_deterioration": float(strong_threshold)}
        case_id = stable_id(definition)
        definitions[case_id] = definition
        daily = run_case(observation, score_obs, order_obs, definition, protocol, protocol["observation_end"])
        rows.append({"case_id": case_id, **definition, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    frame["eligible"] = ((frame["min_year_cumulative_return"] >= 0.0) & (frame["full_linear_annual_proxy"] >= 0.08) & (frame["full_sharpe"] >= 0.5) & (frame["full_max_drawdown"] <= 0.45) & (frame["recent60_linear_annual_proxy"] >= 0.0) & (frame["recent120_linear_annual_proxy"] >= 0.1))
    frame = frame.sort_values(["eligible", "full_linear_annual_proxy", "min_year_cumulative_return", "full_sharpe"], ascending=[False, False, False, False])
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]]
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_this_grid_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
