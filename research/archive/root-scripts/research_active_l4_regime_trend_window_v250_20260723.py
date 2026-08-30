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
import research_active_l4_market_regime_v134_20260722 as v134
import research_active_l4_robust_objective_v56_20260721 as robust
import research_active_l4_score_deterioration_v174_20260722 as v174
import research_active_l4_score_trend_sizing_v234_20260723 as v234
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_regime_trend_window_v250_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v250_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def regime_multiplier(arrays, score, definition, protocol):
    short_definition = {
        **definition,
        "score_trend_lookback": int(definition["weak_lookback"]),
        "score_trend_sensitivity": 1.0,
    }
    long_definition = {
        **definition,
        "score_trend_lookback": int(definition["strong_lookback"]),
        "score_trend_sensitivity": 1.0,
    }
    short = v234.target_multiplier(score, short_definition, protocol)
    long = v234.target_multiplier(score, long_definition, protocol)
    momentum = v134.market_momentum(arrays, int(definition["market_lookback"]))
    use_long = np.isfinite(momentum) & (
        momentum >= float(definition["window_switch_return"])
    )
    return np.where(use_long[:, None], long, short).astype(np.float32)


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
        candidate_target_multiplier_override=regime_multiplier(
            arrays, score, definition, protocol
        ),
        max_positions_override=int(definition["max_positions"]),
    )


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


def definition_for(protocol: dict, threshold: float) -> dict:
    return {
        **protocol["fixed_definition"],
        "weak_lookback": 3,
        "strong_lookback": 5,
        "window_switch_return": float(threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Active L4 市场状态趋势窗口切换研究")
    parser.add_argument("--open-known-2026", action="store_true")
    parser.add_argument("--export-actions", action="store_true")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)

    if args.export_actions:
        if args.threshold is None:
            raise ValueError("导出动作必须显式给出 threshold")
        definition = definition_for(protocol, args.threshold)
        arrays = fresh_arrays()
        score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
        end = str(arrays["dates"][-2])
        daily, actions = run_case(
            arrays, score, order, definition, protocol, end, record_actions=True
        )
        suffix = str(args.threshold).replace("-", "m").replace(".", "p")
        daily_path = OUT / f"daily_local_threshold_{suffix}_through_latest.csv"
        actions_path = OUT / f"replay_actions_threshold_{suffix}_through_latest.csv"
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
        definition_for(protocol, threshold)
        for threshold in protocol["grid"]["window_switch_return"]
    ]
    if args.open_known_2026:
        rows = []
        for definition in definitions:
            for start in protocol["known_stress_buy_starts"]:
                daily = run_case(
                    arrays,
                    score,
                    order,
                    definition,
                    protocol,
                    protocol["known_stress_end"],
                    start,
                )
                rows.append(
                    {
                        "case_id": stable_id(definition),
                        "window_switch_return": definition["window_switch_return"],
                        "buy_start": start,
                        **core.metrics(daily, start, protocol["known_stress_end"]),
                    }
                )
        pd.DataFrame(rows).to_csv(
            OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({"rows": len(rows)}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(arrays, protocol["observation_end"])
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
            "window_switch_return": definition["window_switch_return"],
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
