# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import export_active_l4_v175_latest_research_selection_20260723 as latest_export
import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_adaptive_exit_v162_20260722 as v162
import research_active_l4_robust_objective_v56_20260721 as robust
import research_active_l4_score_deterioration_v174_20260722 as v174
import research_active_l4_score_trend_ensemble_v245_20260723 as v245
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_score_trend_blend_v246_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v246_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def blended_multiplier(score: np.ndarray, definition: dict, protocol: dict) -> np.ndarray:
    ensemble = v245.ensemble_multiplier(score, definition, protocol)
    weight = float(definition["trend_blend_weight"])
    return (1.0 + weight * (ensemble - 1.0)).astype(np.float32)


def run_case(arrays, score, order, definition, protocol, end, start=None, record_actions=False):
    return v162.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start,
        record_actions=record_actions,
        selection_mask_override=v174.selection_mask(
            arrays, definition["max_rank_deterioration"]
        ),
        candidate_target_multiplier_override=blended_multiplier(
            score, definition, protocol
        ),
        max_positions_override=int(definition["max_positions"]),
    )


def fresh_arrays(protocol: dict) -> dict[str, np.ndarray]:
    cache_protocol = json.loads(latest_export.BASE_PROTOCOL.read_text(encoding="utf-8"))
    temp_cache = OUT / "runtime_fresh_cache.npz"
    original_cache = core.CACHE_PATH
    try:
        core.CACHE_PATH = temp_cache
        arrays = core.build_cache(cache_protocol)
    finally:
        core.CACHE_PATH = original_cache
        temp_cache.unlink(missing_ok=True)
    latest_export.populate_latest_signal_row(arrays)
    latest_export.add_liquidity_arrays(arrays)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Active L4 趋势乘数连续混合权重研究")
    parser.add_argument("--open-known-2026", action="store_true")
    parser.add_argument("--export-selected-actions", action="store_true")
    parser.add_argument("--blend-weight", type=float)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(full, 0.0, 7, 0.1)

    if args.export_selected_actions:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        if args.blend_weight is None:
            selected = max(
                frozen["candidates"],
                key=lambda item: item["observation_metrics"]["full_linear_annual_proxy"],
            )
        else:
            selected = next(
                item
                for item in frozen["candidates"]
                if abs(
                    float(item["definition"]["trend_blend_weight"])
                    - float(args.blend_weight)
                )
                < 1e-9
            )
        full = fresh_arrays(protocol)
        score, order = v95.score_pair(full, 0.0, 7, 0.1)
        end = str(full["dates"][-2])
        daily, actions = run_case(
            full,
            score,
            order,
            selected["definition"],
            protocol,
            end,
            record_actions=True,
        )
        suffix = str(selected["definition"]["trend_blend_weight"]).replace(".", "p")
        daily_path = OUT / f"daily_local_blend_{suffix}_through_latest.csv"
        actions_path = OUT / f"replay_actions_blend_{suffix}_through_latest.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        actions.to_csv(actions_path, index=False, encoding="utf-8-sig")
        print(
            json.dumps(
                {
                    "case_id": selected["case_id"],
                    "blend_weight": selected["definition"]["trend_blend_weight"],
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
            definition = item["definition"]
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
                        "trend_blend_weight": definition["trend_blend_weight"],
                        "buy_start": start,
                        **core.metrics(daily, start, protocol["known_stress_end"]),
                    }
                )
        pd.DataFrame(rows).to_csv(
            OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig"
        )
        print(
            json.dumps(
                {"rows": len(rows), "candidates": len(frozen["candidates"])},
                ensure_ascii=False,
            )
        )
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    score_obs = score[: len(observation["dates"])]
    order_obs = order[: len(observation["dates"])]
    rows = []
    candidates = []
    for weight in protocol["grid"]["trend_blend_weight"]:
        definition = {
            **protocol["fixed_definition"],
            "ensemble_id": "observation_top2_equal",
            "components": protocol["fixed_components"],
            "trend_blend_weight": float(weight),
        }
        case_id = stable_id(definition)
        daily = run_case(
            observation,
            score_obs,
            order_obs,
            definition,
            protocol,
            protocol["observation_end"],
        )
        metrics = {
            "case_id": case_id,
            "trend_blend_weight": float(weight),
            **robust.evaluate_robust(daily, protocol),
        }
        rows.append(metrics)
        candidates.append(
            {
                "case_id": case_id,
                "definition": definition,
                "observation_metrics": metrics,
            }
        )
    frame = pd.DataFrame(rows).sort_values("trend_blend_weight")
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    FROZEN.write_text(
        json.dumps(
            {
                "status": "all_preregistered_weights_frozen_before_known_2026_stress",
                "known_2026_used_for_grid_or_selection": False,
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"grid": len(frame), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
