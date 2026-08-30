# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_score_deterioration_v174_20260722 as v174
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "quant/data_file/reports/strategy_agent_active_l4_v260_startup_exposure_v271_20260723"
)
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v271_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def definition_for(protocol: dict, startup_exposure_scale: float) -> dict:
    return {
        **v260.definition_for(protocol, 50),
        "startup_exposure_days": 50,
        "startup_exposure_scale": float(startup_exposure_scale),
    }


def target_multiplier(arrays, score, definition, protocol, start):
    result = v260.v258.v252.warmup_multiplier(
        arrays, score, definition, protocol, start
    )
    days = int(definition["startup_exposure_days"])
    scale = float(definition["startup_exposure_scale"])
    if days <= 0 or abs(scale - 1.0) < 1e-12:
        return result
    dates = arrays["dates"].astype(str)
    start_idx = 0 if start is None else int(np.searchsorted(dates, str(start)))
    stop_idx = min(len(dates), start_idx + days)
    result[start_idx:stop_idx] *= scale
    return result


def run_case(
    arrays,
    score,
    order,
    definition,
    protocol,
    end,
    start=None,
):
    return v260.v258.v252.v162.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start,
        selection_mask_override=v174.selection_mask(
            arrays, definition["max_rank_deterioration"]
        ),
        candidate_target_multiplier_override=target_multiplier(
            arrays, score, definition, protocol, start
        ),
        max_positions_override=int(definition["max_positions"]),
        max_positions_schedule_override=v260.position_schedule(
            arrays, definition, start
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 冷启动仓位倍率研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    definitions = [
        definition_for(protocol, value)
        for value in protocol["grid"]["startup_exposure_scale"]
    ]
    rows = []
    full_rows = []
    frozen = []
    for definition in definitions:
        case_id = stable_id(definition)
        if args.open_known_2026:
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
                        "case_id": case_id,
                        "startup_exposure_scale": definition[
                            "startup_exposure_scale"
                        ],
                        "buy_start": start,
                        **core.metrics(
                            daily, start, protocol["known_stress_end"]
                        ),
                    }
                )
                del daily
                gc.collect()
            continue
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
                "startup_exposure_scale": definition[
                    "startup_exposure_scale"
                ],
                **core.metrics(
                    daily,
                    str(arrays["dates"][0]),
                    protocol["observation_end"],
                ),
            }
        )
        del daily
        gc.collect()
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
                rows.append(
                    {
                        "case_id": case_id,
                        "startup_exposure_scale": definition[
                            "startup_exposure_scale"
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
                del start_daily
                gc.collect()
        frozen.append({"case_id": case_id, "definition": definition})
    if args.open_known_2026:
        pd.DataFrame(rows).to_csv(
            OUT / "known_2026_stress.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(json.dumps({"rows": len(rows)}, ensure_ascii=False))
        return
    pd.DataFrame(full_rows).to_csv(
        OUT / "observation_full_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "development_start_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        frame.groupby("startup_exposure_scale")
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
                "status": "entire_preregistered_grid_frozen_before_known_2026",
                "known_2026_used_for_grid_definition": False,
                "candidates": frozen,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"grid": len(definitions), "development_start_rows": len(frame)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
