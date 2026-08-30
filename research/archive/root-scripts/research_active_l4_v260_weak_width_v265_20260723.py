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
import research_active_l4_market_regime_v134_20260722 as v134
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "quant/data_file/reports/strategy_agent_active_l4_v260_weak_width_v265_20260723"
)
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v265_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def definition_for(protocol: dict, weak_max_positions: int) -> dict:
    return {
        **v260.definition_for(protocol, 50),
        "weak_max_positions": int(weak_max_positions),
        "normal_max_positions": 15,
        "strong_max_positions": 16,
    }


def position_schedule(arrays, definition, start):
    momentum = v134.market_momentum(
        arrays, int(definition["market_lookback"])
    )
    schedule = np.full(
        len(momentum),
        int(definition["weak_max_positions"]),
        dtype=np.int16,
    )
    normal = np.isfinite(momentum) & (
        momentum >= float(definition["market_return_min"])
    )
    strong = np.isfinite(momentum) & (
        momentum >= float(definition["strong_return_min"])
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
    record_actions=False,
):
    return v260.v258.v252.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start,
        record_actions=record_actions,
        max_positions_schedule_override=position_schedule(
            arrays, definition, start
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 弱势持仓宽度研究")
    parser.add_argument("--open-known-2026", action="store_true")
    parser.add_argument("--export-actions", action="store_true")
    parser.add_argument("--weak-max-positions", type=int)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    definitions = [
        definition_for(protocol, value)
        for value in protocol["grid"]["weak_max_positions"]
    ]

    if args.export_actions:
        if args.weak_max_positions is None:
            raise ValueError("导出动作必须明确 weak-max-positions")
        definition = definition_for(protocol, args.weak_max_positions)
        arrays = v260.v258.v252.fresh_arrays()
        score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
        end = str(arrays["dates"][-2])
        daily, actions = run_case(
            arrays,
            score,
            order,
            definition,
            protocol,
            end,
            record_actions=True,
        )
        suffix = f"weak{args.weak_max_positions}"
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
                        "weak_max_positions": definition[
                            "weak_max_positions"
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
                "weak_max_positions": definition["weak_max_positions"],
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
                        "weak_max_positions": definition[
                            "weak_max_positions"
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
        frame.groupby("weak_max_positions")
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
