# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from . import production_v260_active_l4_v175_latest_research_selection_20260723 as latest_export
from . import production_v260_active_l4_10d_smoothing_refine_v95_20260722 as v95
from . import production_v260_active_l4_adaptive_exit_v162_20260722 as v162
from . import production_v260_active_l4_robust_objective_v56_20260721 as robust
from . import production_v260_active_l4_score_deterioration_v174_20260722 as v174
from . import production_v260_active_l4_score_trend_sizing_v234_20260723 as v234
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_trend_exposure_surface_v247_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v247_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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
        candidate_target_multiplier_override=v234.target_multiplier(
            score, definition, protocol
        ),
        max_positions_override=int(definition["max_positions"]),
    )


def definition_for(protocol: dict, lookback: int, sensitivity: float, exposure: float) -> dict:
    return {
        **protocol["fixed_definition"],
        "score_trend_lookback": int(lookback),
        "score_trend_sensitivity": float(sensitivity),
        "exposure_scale": float(exposure),
        "normal_target_pct": 0.25 * float(exposure),
        "strong_target_pct": 0.375 * float(exposure),
    }


def fresh_arrays() -> dict[str, np.ndarray]:
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
    parser = argparse.ArgumentParser(description="Active L4 趋势与仓位尺度二维曲面")
    parser.add_argument("--open-known-2026", action="store_true")
    parser.add_argument("--export-actions", action="store_true")
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--sensitivity", type=float)
    parser.add_argument("--exposure", type=float)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(full, 0.0, 7, 0.1)

    if args.export_actions:
        if args.lookback is None or args.sensitivity is None or args.exposure is None:
            raise ValueError("导出动作必须显式给出 lookback/sensitivity/exposure")
        definition = definition_for(
            protocol, args.lookback, args.sensitivity, args.exposure
        )
        full = fresh_arrays()
        score, order = v95.score_pair(full, 0.0, 7, 0.1)
        end = str(full["dates"][-2])
        daily, actions = run_case(
            full, score, order, definition, protocol, end, record_actions=True
        )
        suffix = "l{}_s{}_e{}".format(
            args.lookback,
            str(args.sensitivity).replace(".", "p"),
            str(args.exposure).replace(".", "p"),
        )
        daily_path = OUT / f"daily_local_{suffix}_through_latest.csv"
        actions_path = OUT / f"replay_actions_{suffix}_through_latest.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        actions.to_csv(actions_path, index=False, encoding="utf-8-sig")
        print(
            json.dumps(
                {
                    "case_id": stable_id(definition),
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

    definitions = [
        definition_for(protocol, lookback, sensitivity, exposure)
        for lookback, sensitivity, exposure in product(
            protocol["grid"]["score_trend_lookback"],
            protocol["grid"]["score_trend_sensitivity"],
            protocol["grid"]["exposure_scale"],
        )
    ]

    if args.open_known_2026:
        rows = []
        for definition in definitions:
            case_id = stable_id(definition)
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
                        "case_id": case_id,
                        "score_trend_lookback": definition["score_trend_lookback"],
                        "score_trend_sensitivity": definition["score_trend_sensitivity"],
                        "exposure_scale": definition["exposure_scale"],
                        "buy_start": start,
                        **core.metrics(daily, start, protocol["known_stress_end"]),
                    }
                )
        pd.DataFrame(rows).to_csv(
            OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({"rows": len(rows), "grid": len(definitions)}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    score_obs = score[: len(observation["dates"])]
    order_obs = order[: len(observation["dates"])]
    rows = []
    frozen = []
    for definition in definitions:
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
            "score_trend_lookback": definition["score_trend_lookback"],
            "score_trend_sensitivity": definition["score_trend_sensitivity"],
            "exposure_scale": definition["exposure_scale"],
            **robust.evaluate_robust(daily, protocol),
        }
        rows.append(metrics)
        frozen.append(
            {
                "case_id": case_id,
                "definition": definition,
                "observation_metrics": metrics,
            }
        )
    pd.DataFrame(rows).to_csv(
        OUT / "observation_grid.csv", index=False, encoding="utf-8-sig"
    )
    FROZEN.write_text(
        json.dumps(
            {
                "status": "entire_preregistered_grid_frozen_before_known_2026",
                "known_2026_used_for_grid_definition": False,
                "candidates": frozen,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"grid": len(rows), "frozen": len(frozen)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
