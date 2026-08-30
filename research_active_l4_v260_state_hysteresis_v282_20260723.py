# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_adaptive_exit_v162_20260722 as v162
import research_active_l4_breadth_exit_v109_20260722 as v109
import research_active_l4_cold_start_trend_warmup_v252_20260723 as v252
import research_active_l4_market_regime_v134_20260722 as v134
import research_active_l4_score_deterioration_v174_20260722 as v174
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "quant/data_file/reports/strategy_agent_active_l4_v260_state_hysteresis_v282_20260723"
)
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v282_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def definition_for(protocol: dict, band: float) -> dict:
    return {
        **v260.definition_for(protocol, protocol["position_warmup_days"]),
        "state_hysteresis_band": float(band),
    }


def state_labels(momentum: np.ndarray, definition: dict) -> np.ndarray:
    normal_min = float(definition["market_return_min"])
    strong_min = float(definition["strong_return_min"])
    band = float(definition["state_hysteresis_band"])
    labels = np.zeros(len(momentum), dtype=np.int8)
    if band <= 0:
        labels[np.isfinite(momentum) & (momentum >= normal_min)] = 1
        labels[np.isfinite(momentum) & (momentum >= strong_min)] = 2
        return labels

    state = 0
    for idx, value in enumerate(momentum):
        if not np.isfinite(value):
            labels[idx] = state
            continue
        if state == 0:
            if value >= strong_min + band:
                state = 2
            elif value >= normal_min + band:
                state = 1
        elif state == 1:
            if value >= strong_min + band:
                state = 2
            elif value < normal_min - band:
                state = 0
        else:
            if value < normal_min - band:
                state = 0
            elif value < strong_min - band:
                state = 1
        labels[idx] = state
    return labels


def min_hold_schedule(momentum: np.ndarray, band: float) -> np.ndarray:
    result = np.full(len(momentum), 4, dtype=np.int16)
    if band <= 0:
        result[np.isfinite(momentum) & (momentum >= 0.04)] = 3
        return result

    strong = False
    for idx, value in enumerate(momentum):
        if np.isfinite(value):
            if strong and value < 0.04 - band:
                strong = False
            elif not strong and value >= 0.04 + band:
                strong = True
        result[idx] = 3 if strong else 4
    return result


def state_schedules(arrays, definition, start):
    momentum = v134.market_momentum(
        arrays,
        int(definition["market_lookback"]),
    )
    labels = state_labels(momentum, definition)

    gross = np.full(len(labels), float(definition["low_gross"]), dtype=float)
    gross[labels >= 1] = 1.0

    target = np.full(
        len(labels),
        float(definition["weak_target_pct"]),
        dtype=float,
    )
    target[labels == 1] = float(definition["normal_target_pct"])
    target[labels == 2] = float(definition["strong_target_pct"])

    max_positions = np.full(
        len(labels),
        int(definition["weak_max_positions"]),
        dtype=np.int16,
    )
    max_positions[labels == 1] = int(definition["normal_max_positions"])
    max_positions[labels == 2] = int(definition["strong_max_positions"])

    dates = arrays["dates"].astype(str)
    start_idx = 0 if start is None else int(np.searchsorted(dates, str(start)))
    stop_idx = min(
        len(dates),
        start_idx + int(definition["position_warmup_days"]),
    )
    max_positions[start_idx:stop_idx] = int(definition["max_positions"])
    return (
        gross,
        target,
        min_hold_schedule(
            momentum,
            float(definition["state_hysteresis_band"]),
        ),
        max_positions,
    )


def run_case(arrays, score, order, definition, protocol, end, start=None):
    gross, target, min_hold, max_positions = state_schedules(
        arrays,
        definition,
        start,
    )
    return v109.simulate(
        arrays,
        score,
        order,
        definition,
        v162.case_protocol(protocol, definition),
        end,
        start,
        gross_target_override=gross,
        target_cohorts_override=4,
        target_pct_override=target,
        min_hold_days_override=min_hold,
        selection_mask_override=v174.selection_mask(
            arrays,
            definition["max_rank_deterioration"],
        ),
        candidate_target_multiplier_override=v252.warmup_multiplier(
            arrays,
            score,
            definition,
            protocol,
            start,
        ),
        max_positions_override=int(definition["max_positions"]),
        max_positions_schedule_override=max_positions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 市场状态滞回带研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    definitions = [
        definition_for(protocol, value)
        for value in protocol["grid"]["state_hysteresis_band"]
    ]

    if args.open_known_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        if frozen["status"] != "entire_grid_frozen_before_known_2026":
            raise RuntimeError("冻结文件状态不允许打开2026压力窗口")
        rows = []
        for item in frozen["candidates"]:
            definition = item["definition"]
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
                        "case_id": item["case_id"],
                        "state_hysteresis_band": definition[
                            "state_hysteresis_band"
                        ],
                        "buy_start": start,
                        **core.metrics(
                            daily,
                            start,
                            protocol["known_stress_end"],
                        ),
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(
            OUT / "known_2026_stress.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (
            frame.groupby("state_hysteresis_band")
            .agg(
                starts=("buy_start", "count"),
                min_linear_annual_proxy=("linear_annual_proxy", "min"),
                median_linear_annual_proxy=("linear_annual_proxy", "median"),
                mean_linear_annual_proxy=("linear_annual_proxy", "mean"),
                min_sharpe=("sharpe", "min"),
                max_drawdown=("max_drawdown", "max"),
            )
            .reset_index()
            .to_csv(
                OUT / "known_2026_stress_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
        )
        print(json.dumps({"rows": len(frame)}, ensure_ascii=False))
        return

    full_rows = []
    start_rows = []
    frozen = []
    for definition in definitions:
        case_id = stable_id(definition)
        daily = run_case(
            arrays,
            score,
            order,
            definition,
            protocol,
            protocol["observation_end"],
        )
        full_rows.append(
            {
                "case_id": case_id,
                "state_hysteresis_band": definition[
                    "state_hysteresis_band"
                ],
                **core.metrics(
                    daily,
                    str(arrays["dates"][0]),
                    protocol["observation_end"],
                ),
            }
        )
        for anchor, starts in protocol["development_start_groups"].items():
            for start in starts:
                start_daily = run_case(
                    arrays,
                    score,
                    order,
                    definition,
                    protocol,
                    protocol["observation_end"],
                    start,
                )
                start_rows.append(
                    {
                        "case_id": case_id,
                        "state_hysteresis_band": definition[
                            "state_hysteresis_band"
                        ],
                        "anchor": anchor,
                        "buy_start": start,
                        **core.metrics(
                            start_daily,
                            start,
                            protocol["observation_end"],
                        ),
                    }
                )
        frozen.append({"case_id": case_id, "definition": definition})

    pd.DataFrame(full_rows).to_csv(
        OUT / "observation_full_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    frame = pd.DataFrame(start_rows)
    frame.to_csv(
        OUT / "development_start_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        frame.groupby("state_hysteresis_band")
        .agg(
            starts=("buy_start", "count"),
            min_linear_annual_proxy=("linear_annual_proxy", "min"),
            median_linear_annual_proxy=("linear_annual_proxy", "median"),
            mean_linear_annual_proxy=("linear_annual_proxy", "mean"),
            min_sharpe=("sharpe", "min"),
            max_drawdown=("max_drawdown", "max"),
        )
        .reset_index()
        .to_csv(
            OUT / "development_start_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )
    FROZEN.write_text(
        json.dumps(
            {
                "status": "entire_grid_frozen_before_known_2026",
                "known_2026_used_for_grid_definition": True,
                "known_2026_is_independent_validation": False,
                "candidates": frozen,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "grid": len(definitions),
                "development_start_rows": len(frame),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
