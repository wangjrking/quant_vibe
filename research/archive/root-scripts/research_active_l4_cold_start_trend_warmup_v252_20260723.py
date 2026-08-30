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
import research_active_l4_score_trend_sizing_v234_20260723 as v234
import research_active_l4_trend_exposure_surface_v247_20260723 as v247
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_cold_start_trend_warmup_v252_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v252_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def warmup_multiplier(arrays, score, definition, protocol, start):
    steady = {
        **definition,
        "score_trend_lookback": int(definition["steady_lookback"]),
        "score_trend_sensitivity": 1.0,
    }
    warm = {
        **definition,
        "score_trend_lookback": int(definition["warmup_lookback"]),
        "score_trend_sensitivity": 1.0,
    }
    result = v234.target_multiplier(score, steady, protocol)
    warmup_days = int(definition["warmup_days"])
    if warmup_days <= 0:
        return result
    dates = arrays["dates"].astype(str)
    start_idx = 0 if start is None else int(np.searchsorted(dates, str(start)))
    stop_idx = min(len(dates), start_idx + warmup_days)
    warm_values = v234.target_multiplier(score, warm, protocol)
    result[start_idx:stop_idx] = warm_values[start_idx:stop_idx]
    return result


def run_case(
    arrays,
    score,
    order,
    definition,
    protocol,
    end,
    start=None,
    record_actions=False,
    entry_slots_schedule_override=None,
    max_positions_schedule_override=None,
):
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
        candidate_target_multiplier_override=warmup_multiplier(
            arrays, score, definition, protocol, start
        ),
        max_positions_override=int(definition["max_positions"]),
        max_positions_schedule_override=max_positions_schedule_override,
        entry_slots_schedule_override=entry_slots_schedule_override,
    )


def definition_for(protocol: dict, warmup_days: int) -> dict:
    return {
        **protocol["fixed_definition"],
        "warmup_lookback": 3,
        "steady_lookback": 5,
        "warmup_days": int(warmup_days),
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
    parser = argparse.ArgumentParser(description="Active L4 冷启动趋势窗口预热研究")
    parser.add_argument("--open-known-2026", action="store_true")
    parser.add_argument("--export-actions", action="store_true")
    parser.add_argument("--warmup-days", type=int)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)

    if args.export_actions:
        if args.warmup_days is None:
            raise ValueError("导出动作必须显式给出 warmup-days")
        definition = definition_for(protocol, args.warmup_days)
        arrays = fresh_arrays()
        score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
        end = str(arrays["dates"][-2])
        daily, actions = run_case(
            arrays, score, order, definition, protocol, end, record_actions=True
        )
        daily_path = OUT / f"daily_local_warmup_{args.warmup_days}_through_latest.csv"
        actions_path = OUT / f"replay_actions_warmup_{args.warmup_days}_through_latest.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        actions.to_csv(actions_path, index=False, encoding="utf-8-sig")
        print(
            json.dumps(
                {
                    "case_id": stable_id(definition),
                    "warmup_days": args.warmup_days,
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
        definition_for(protocol, days) for days in protocol["grid"]["warmup_days"]
    ]
    if args.open_known_2026:
        rows = []
        for definition in definitions:
            for start in protocol["known_stress_starts"]:
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
                        "warmup_days": definition["warmup_days"],
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
    full_rows = []
    start_rows = []
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
        full_metrics = {
            "case_id": case_id,
            "warmup_days": definition["warmup_days"],
            **robust.evaluate_robust(daily, protocol),
        }
        full_rows.append(full_metrics)
        for anchor, starts in protocol["development_start_groups"].items():
            for start in starts:
                start_daily = run_case(
                    observation,
                    score_obs,
                    order_obs,
                    definition,
                    protocol,
                    protocol["observation_end"],
                    start,
                )
                start_rows.append(
                    {
                        "case_id": case_id,
                        "warmup_days": definition["warmup_days"],
                        "anchor": anchor,
                        "buy_start": start,
                        **core.metrics(
                            start_daily, start, protocol["observation_end"]
                        ),
                    }
                )
        frozen.append(
            {
                "case_id": case_id,
                "definition": definition,
                "observation_full_metrics": full_metrics,
            }
        )
    pd.DataFrame(full_rows).to_csv(
        OUT / "observation_full_grid.csv", index=False, encoding="utf-8-sig"
    )
    start_frame = pd.DataFrame(start_rows)
    start_frame.to_csv(
        OUT / "development_start_results.csv", index=False, encoding="utf-8-sig"
    )
    summary = (
        start_frame.groupby("warmup_days")
        .agg(
            starts=("buy_start", "count"),
            min_linear_annual_proxy=("linear_annual_proxy", "min"),
            median_linear_annual_proxy=("linear_annual_proxy", "median"),
            mean_linear_annual_proxy=("linear_annual_proxy", "mean"),
            min_sharpe=("sharpe", "min"),
            max_drawdown=("max_drawdown", "max"),
        )
        .reset_index()
    )
    summary.to_csv(
        OUT / "development_start_summary.csv", index=False, encoding="utf-8-sig"
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
    print(
        json.dumps(
            {
                "grid": len(full_rows),
                "development_start_rows": len(start_frame),
                "frozen": len(frozen),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
