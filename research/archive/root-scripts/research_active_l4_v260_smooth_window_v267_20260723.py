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
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "quant/data_file/reports/strategy_agent_active_l4_v260_smooth_window_v267_20260723"
)
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v267_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def definition_for(protocol: dict, smooth_window: int) -> dict:
    return {
        **v260.definition_for(protocol, 50),
        "smooth_window": int(smooth_window),
        "current_weight": 0.1,
    }


def score_pair(arrays, definition):
    return v95.score_pair(
        arrays,
        0.0,
        int(definition["smooth_window"]),
        float(definition["current_weight"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 分数平滑窗口邻域研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    definitions = [
        definition_for(protocol, value)
        for value in protocol["grid"]["smooth_window"]
    ]
    rows = []
    full_rows = []
    frozen = []
    for definition in definitions:
        score, order = score_pair(arrays, definition)
        case_id = stable_id(definition)
        if args.open_known_2026:
            for start in protocol["known_stress_starts"]:
                daily = v260.run_case(
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
                        "smooth_window": definition["smooth_window"],
                        "buy_start": start,
                        **core.metrics(
                            daily, start, protocol["known_stress_end"]
                        ),
                    }
                )
                del daily
                gc.collect()
            del score, order
            gc.collect()
            continue
        daily = v260.run_case(
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
                "smooth_window": definition["smooth_window"],
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
                    protocol,
                    protocol["observation_end"],
                    start,
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "smooth_window": definition["smooth_window"],
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
        del score, order
        gc.collect()

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
        frame.groupby("smooth_window")
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
