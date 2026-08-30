# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_max_positions_v195_20260723 as v195
import research_active_l4_rank_smoothing_v94_20260722 as v94
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_score_quantile_v241_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v241_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def rolling_nanquantile(values: np.ndarray, window: int, quantile: float) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float32)
    for end in range(1, len(values) + 1):
        start = max(0, end - window)
        block = values[start:end]
        finite_count = np.isfinite(block).sum(axis=0)
        ordered = np.sort(block, axis=0)
        position = np.maximum(finite_count - 1, 0) * quantile
        lower = np.floor(position).astype(np.intp)
        upper = np.ceil(position).astype(np.intp)
        columns = np.arange(values.shape[1], dtype=np.intp)
        lower_value = ordered[lower, columns]
        upper_value = ordered[upper, columns]
        interpolated = lower_value + (upper_value - lower_value) * (position - lower)
        interpolated[finite_count == 0] = np.nan
        result[end - 1] = interpolated.astype(np.float32)
    return result


def score_pair(arrays: dict, aggregator: str) -> tuple[np.ndarray, np.ndarray]:
    raw = arrays["rank_10d"].astype(np.float32)
    if aggregator == "mean":
        history = v94.rolling_nanmean(raw, 7)
    else:
        quantile = float(aggregator.split("_", 1)[1])
        history = rolling_nanquantile(raw, 7, quantile)
    score = (0.1 * raw + 0.9 * history).astype(np.float32)
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")
    return score, order


def run_case(arrays, definition, protocol, end, start=None):
    score, order = score_pair(arrays, definition["history_aggregator"])
    return v195.run_case(arrays, score, order, definition, protocol, end, start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Active L4 七日分位数稳健排序研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}

    if args.open_known_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        score_cache = {}
        for item in frozen["candidates"]:
            definition = item["definition"]
            aggregator = definition["history_aggregator"]
            if aggregator not in score_cache:
                score_cache[aggregator] = score_pair(full, aggregator)
            score, order = score_cache[aggregator]
            for start in protocol["known_stress_buy_starts"]:
                daily = v195.run_case(
                    full, score, order, definition, protocol, protocol["known_stress_end"], start
                )
                rows.append(
                    {
                        "case_id": item["case_id"],
                        "history_aggregator": aggregator,
                        "buy_start": start,
                        **core.metrics(daily, start, protocol["known_stress_end"]),
                    }
                )
        pd.DataFrame(rows).to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({"rows": len(rows), "candidates": len(frozen["candidates"])}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    rows, definitions = [], {}
    for aggregator in protocol["grid"]["history_aggregator"]:
        definition = {**protocol["fixed_definition"], "history_aggregator": aggregator}
        case_id = stable_id(definition)
        definitions[case_id] = definition
        daily = run_case(observation, definition, protocol, protocol["observation_end"])
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
    print(
        json.dumps(
            {"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
