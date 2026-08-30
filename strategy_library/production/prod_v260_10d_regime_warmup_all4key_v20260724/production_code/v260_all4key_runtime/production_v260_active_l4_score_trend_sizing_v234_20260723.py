# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import production_v260_active_l4_10d_smoothing_refine_v95_20260722 as v95
from . import production_v260_active_l4_adaptive_exit_v162_20260722 as v162
from . import production_v260_active_l4_robust_objective_v56_20260721 as robust
from . import production_v260_active_l4_score_deterioration_v174_20260722 as v174
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core
from . import production_v260_active_l4_v175_latest_research_selection_20260723 as latest_export


ROOT = Path(__file__).resolve().parents[7]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_score_trend_sizing_v234_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v234_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def target_multiplier(score: np.ndarray, definition: dict, protocol: dict) -> np.ndarray:
    lookback = int(definition["score_trend_lookback"])
    sensitivity = float(definition["score_trend_sensitivity"])
    result = np.ones(score.shape, dtype=np.float32)
    if sensitivity <= 0:
        return result
    delta = np.full(score.shape, np.nan, dtype=np.float32)
    delta[lookback:] = score[lookback:] - score[:-lookback]
    valid = np.isfinite(delta)
    result[valid] = np.clip(
        1.0 + sensitivity * delta[valid],
        float(protocol["target_multiplier_floor"]),
        float(protocol["target_multiplier_cap"]),
    )
    return result


def run_case(arrays, score, order, definition, protocol, end, start=None):
    return v162.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start,
        selection_mask_override=v174.selection_mask(arrays, definition["max_rank_deterioration"]),
        candidate_target_multiplier_override=target_multiplier(score, definition, protocol),
        max_positions_override=int(definition["max_positions"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Active L4 同股分数趋势连续仓位研究")
    parser.add_argument("--open-known-2026", action="store_true")
    parser.add_argument("--export-selected-actions", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(full, 0.0, 7, 0.1)

    if args.export_selected_actions:
        cache_protocol = json.loads(latest_export.BASE_PROTOCOL.read_text(encoding="utf-8"))
        temp_cache = OUT / "runtime_fresh_cache.npz"
        original_cache = core.CACHE_PATH
        try:
            core.CACHE_PATH = temp_cache
            full = core.build_cache(cache_protocol)
        finally:
            core.CACHE_PATH = original_cache
            temp_cache.unlink(missing_ok=True)
        latest_export.populate_latest_signal_row(full)
        latest_export.add_liquidity_arrays(full)
        score, order = v95.score_pair(full, 0.0, 7, 0.1)
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        selected = next(
            item
            for item in frozen["candidates"]
            if int(item["definition"]["score_trend_lookback"]) == 5
            and abs(float(item["definition"]["score_trend_sensitivity"]) - 1.0) < 1e-12
        )
        end = str(full["dates"][-2])
        daily, actions = v162.run_case(
            full,
            score,
            order,
            selected["definition"],
            protocol,
            end,
            record_actions=True,
            selection_mask_override=v174.selection_mask(
                full, selected["definition"]["max_rank_deterioration"]
            ),
            candidate_target_multiplier_override=target_multiplier(
                score, selected["definition"], protocol
            ),
            max_positions_override=int(selected["definition"]["max_positions"]),
        )
        daily_path = OUT / "daily_local_lookback5_sensitivity1_through_latest.csv"
        actions_path = OUT / "replay_actions_lookback5_sensitivity1_through_latest.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        actions.to_csv(actions_path, index=False, encoding="utf-8-sig")
        print(
            json.dumps(
                {
                    "case_id": selected["case_id"],
                    "end": end,
                    "daily_rows": len(daily),
                    "action_rows": len(actions),
                    "daily_path": str(daily_path),
                    "actions_path": str(actions_path),
                },
                ensure_ascii=False,
            )
        )
        return

    if args.open_known_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for item in frozen["candidates"]:
            for start in protocol["known_stress_buy_starts"]:
                daily = run_case(
                    full,
                    score,
                    order,
                    item["definition"],
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
        pd.DataFrame(rows).to_csv(
            OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({"rows": len(rows), "candidates": len(frozen["candidates"])}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    score_obs = score[: len(observation["dates"])]
    order_obs = order[: len(observation["dates"])]
    rows, definitions = [], {}
    for item in protocol["grid"]:
        definition = {**protocol["fixed_definition"], **item}
        case_id = stable_id(definition)
        definitions[case_id] = definition
        daily = run_case(
            observation,
            score_obs,
            order_obs,
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
    print(
        json.dumps(
            {"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
