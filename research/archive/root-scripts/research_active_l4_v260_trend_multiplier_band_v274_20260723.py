# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "quant/data_file/reports/strategy_agent_active_l4_v260_trend_multiplier_band_v274_20260723"
)
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v274_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def case_parts(protocol: dict, band: float):
    case_protocol = copy.deepcopy(protocol)
    case_protocol["target_multiplier_floor"] = 1.0 - float(band)
    case_protocol["target_multiplier_cap"] = 1.0 + float(band)
    definition = {
        **v260.definition_for(case_protocol, 50),
        "target_multiplier_band": float(band),
    }
    return case_protocol, definition


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 分数趋势仓位带宽研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    cases = [
        case_parts(protocol, value)
        for value in protocol["grid"]["target_multiplier_band"]
    ]
    rows = []
    full_rows = []
    frozen = []
    for case_protocol, definition in cases:
        case_id = stable_id(definition)
        band = definition["target_multiplier_band"]
        if args.open_known_2026:
            for start in protocol["known_stress_starts"]:
                daily = v260.run_case(
                    arrays,
                    score,
                    order,
                    definition,
                    case_protocol,
                    protocol["known_stress_end"],
                    start,
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "target_multiplier_band": band,
                        "buy_start": start,
                        **core.metrics(
                            daily, start, protocol["known_stress_end"]
                        ),
                    }
                )
                del daily
                gc.collect()
            continue
        daily = v260.run_case(
            arrays,
            score,
            order,
            definition,
            case_protocol,
            protocol["observation_end"],
        )
        full_rows.append(
            {
                "case_id": case_id,
                "target_multiplier_band": band,
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
                start_daily = v260.run_case(
                    arrays,
                    score,
                    order,
                    definition,
                    case_protocol,
                    protocol["observation_end"],
                    start,
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "target_multiplier_band": band,
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
        frozen.append(
            {
                "case_id": case_id,
                "definition": definition,
                "target_multiplier_floor": case_protocol[
                    "target_multiplier_floor"
                ],
                "target_multiplier_cap": case_protocol[
                    "target_multiplier_cap"
                ],
            }
        )
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
        frame.groupby("target_multiplier_band")
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
            {"grid": len(cases), "development_start_rows": len(frame)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
