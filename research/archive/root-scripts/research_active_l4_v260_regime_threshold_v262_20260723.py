# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_market_regime_v134_20260722 as v134
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "quant/data_file/reports/strategy_agent_active_l4_v260_regime_threshold_v262_20260723"
)
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v262_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def definition_for(protocol: dict, weak_threshold: float, strong_threshold: float):
    return {
        **v260.definition_for(protocol, 50),
        "position_weak_threshold": float(weak_threshold),
        "position_strong_threshold": float(strong_threshold),
    }


def position_schedule(arrays, definition, start):
    momentum = v134.market_momentum(
        arrays, int(definition["market_lookback"])
    )
    schedule = np.full(len(momentum), 14, dtype=np.int16)
    normal = np.isfinite(momentum) & (
        momentum >= float(definition["position_weak_threshold"])
    )
    strong = np.isfinite(momentum) & (
        momentum >= float(definition["position_strong_threshold"])
    )
    schedule[normal] = 15
    schedule[strong] = 16
    dates = arrays["dates"].astype(str)
    start_idx = 0 if start is None else int(np.searchsorted(dates, str(start)))
    stop_idx = min(
        len(schedule),
        start_idx + int(definition["position_warmup_days"]),
    )
    schedule[start_idx:stop_idx] = 16
    return schedule


def run_case(
    arrays,
    score,
    order,
    definition,
    protocol,
    end,
    start=None,
):
    return v260.v258.v252.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start,
        max_positions_schedule_override=position_schedule(
            arrays, definition, start
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 状态持仓阈值邻域研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    definitions = [
        definition_for(protocol, weak, strong)
        for weak, strong in product(
            protocol["grid"]["position_weak_threshold"],
            protocol["grid"]["position_strong_threshold"],
        )
        if weak < strong
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
                        "position_weak_threshold": definition[
                            "position_weak_threshold"
                        ],
                        "position_strong_threshold": definition[
                            "position_strong_threshold"
                        ],
                        "buy_start": start,
                        **core.metrics(
                            daily, start, protocol["known_stress_end"]
                        ),
                    }
                )
                del daily
                gc.collect()
        pd.DataFrame(rows).to_csv(
            OUT / "known_2026_stress.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(json.dumps({"rows": len(rows)}, ensure_ascii=False))
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
                "position_weak_threshold": definition[
                    "position_weak_threshold"
                ],
                "position_strong_threshold": definition[
                    "position_strong_threshold"
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
                        "position_weak_threshold": definition[
                            "position_weak_threshold"
                        ],
                        "position_strong_threshold": definition[
                            "position_strong_threshold"
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
        del daily
        gc.collect()
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
        frame.groupby(
            ["position_weak_threshold", "position_strong_threshold"]
        )
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
