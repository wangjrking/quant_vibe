# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_adaptive_exit_v162_20260722 as v162
import research_active_l4_cold_start_trend_warmup_v252_20260723 as v252
import research_active_l4_score_deterioration_v174_20260722 as v174
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "quant/data_file/reports/strategy_agent_active_l4_v260_rebalance_breadth_v278_20260723"
)
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v278_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def definitions(protocol: dict) -> list[dict]:
    base = v260.definition_for(protocol, protocol["position_warmup_days"])
    return [
        {
            **base,
            "daily_entry_slots": int(daily_entry_slots),
            "max_daily_score_sells": int(max_daily_score_sells),
        }
        for daily_entry_slots, max_daily_score_sells in product(
            protocol["grid"]["daily_entry_slots"],
            protocol["grid"]["max_daily_score_sells"],
        )
    ]


def run_case(arrays, score, order, definition, protocol, end, start=None):
    slots = np.full(
        len(arrays["dates"]),
        int(definition["daily_entry_slots"]),
        dtype=np.int16,
    )
    sell_cap = int(definition["max_daily_score_sells"])
    return v162.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start,
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
        max_positions_schedule_override=v260.position_schedule(
            arrays,
            definition,
            start,
        ),
        entry_slots_schedule_override=slots,
        max_daily_score_sells_override=None if sell_cap >= 999 else sell_cap,
        split_candidate_target_override=True,
    )


def metric_row(definition: dict) -> dict:
    return {
        "daily_entry_slots": definition["daily_entry_slots"],
        "max_daily_score_sells": definition["max_daily_score_sells"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 每日新增宽度与卖出上限邻域")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)

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
                        **metric_row(definition),
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
            frame.groupby(["daily_entry_slots", "max_daily_score_sells"])
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
    for definition in definitions(protocol):
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
                **metric_row(definition),
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
                        **metric_row(definition),
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
        frame.groupby(["daily_entry_slots", "max_daily_score_sells"])
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
            {
                "grid": len(frozen),
                "development_start_rows": len(frame),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
