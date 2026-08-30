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
import research_active_l4_adaptive_exit_v162_20260722 as v162
import research_active_l4_robust_objective_v56_20260721 as robust
import research_active_l4_score_deterioration_v174_20260722 as v174
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_liquidity_score_v221_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v221_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def amount_percentile(amount: np.ndarray) -> np.ndarray:
    return pd.DataFrame(amount).rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float32)


def adjusted_score(base_score: np.ndarray, amount_rank: np.ndarray, weight: float) -> tuple[np.ndarray, np.ndarray]:
    score = ((1.0 - weight) * base_score + weight * amount_rank).astype(np.float32)
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")
    return score, order


def run_case(arrays, score, order, definition, protocol, end, start=None):
    local_protocol = copy.deepcopy(protocol)
    local_protocol["fixed_universe"]["amount_min"] = int(definition["amount_min"])
    return v162.run_case(
        arrays,
        score,
        order,
        definition,
        local_protocol,
        end,
        start,
        selection_mask_override=v174.selection_mask(arrays, definition["max_rank_deterioration"]),
        max_positions_override=int(definition["max_positions"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Active L4 连续流动性评分研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    base_score, _ = v95.score_pair(full, 0.0, 7, 0.1)
    amount_rank = amount_percentile(full["amount"])
    score_cache = {
        float(weight): adjusted_score(base_score, amount_rank, float(weight))
        for weight in protocol["grid"]["liquidity_weight"]
    }

    if args.open_known_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for item in frozen["candidates"]:
            definition = item["definition"]
            score, order = score_cache[float(definition["liquidity_weight"])]
            for start in protocol["known_stress_buy_starts"]:
                daily = run_case(
                    full,
                    score,
                    order,
                    definition,
                    protocol,
                    protocol["known_stress_end"],
                    start,
                )
                rows.append(
                    {
                        "case_id": item["case_id"],
                        "buy_start": start,
                        **core.metrics(daily, start, protocol["known_stress_end"]),
                    }
                )
        pd.DataFrame(rows).to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({"rows": len(rows), "candidates": len(frozen["candidates"])}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    rows, definitions = [], {}
    for amount_min, liquidity_weight in product(
        protocol["grid"]["amount_min"],
        protocol["grid"]["liquidity_weight"],
    ):
        definition = {
            **protocol["fixed_definition"],
            "amount_min": int(amount_min),
            "liquidity_weight": float(liquidity_weight),
        }
        case_id = stable_id(definition)
        definitions[case_id] = definition
        score, order = score_cache[float(liquidity_weight)]
        daily = run_case(
            observation,
            score[: len(observation["dates"])],
            order[: len(observation["dates"])],
            definition,
            protocol,
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
    selected = frame[frame["eligible"]]
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
                "status": "frozen_before_known_2026_stress",
                "known_2026_used_for_this_grid_selection": False,
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
