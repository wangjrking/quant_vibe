# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import export_active_l4_v175_latest_research_selection_20260723 as latest_export
import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_cold_start_trend_warmup_v252_20260723 as v252
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_cold_start_fill_v256_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_grid_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v256_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def entry_schedule(arrays, start, startup_days: int, startup_slots: int):
    schedule = np.ones(len(arrays["dates"]), dtype=np.int16)
    if startup_days <= 0 or startup_slots <= 1:
        return schedule
    dates = arrays["dates"].astype(str)
    start_idx = 0 if start is None else int(np.searchsorted(dates, str(start)))
    stop_idx = min(len(dates), start_idx + int(startup_days))
    schedule[start_idx:stop_idx] = int(startup_slots)
    return schedule


def run_case(arrays, score, order, definition, protocol, end, start=None, record_actions=False):
    return v252.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start,
        record_actions=record_actions,
        entry_slots_schedule_override=entry_schedule(
            arrays,
            start,
            int(definition["startup_days"]),
            int(definition["startup_slots"]),
        ),
    )


def definition_for(protocol: dict, days: int, slots: int) -> dict:
    return {
        **v252.definition_for(protocol, 50),
        "startup_days": int(days),
        "startup_slots": int(slots),
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
    parser = argparse.ArgumentParser(description="Active L4 冷启动加速建仓研究")
    parser.add_argument("--open-known-2026", action="store_true")
    parser.add_argument("--export-actions", action="store_true")
    parser.add_argument("--startup-days", type=int)
    parser.add_argument("--startup-slots", type=int)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    pairs = [(0, 1)] + list(
        product(
            protocol["grid"]["startup_days"],
            protocol["grid"]["startup_slots"],
        )
    )
    definitions = [
        definition_for(protocol, int(days), int(slots)) for days, slots in pairs
    ]

    if args.export_actions:
        if args.startup_days is None or args.startup_slots is None:
            raise ValueError("导出动作必须给出 startup-days 和 startup-slots")
        definition = definition_for(
            protocol, args.startup_days, args.startup_slots
        )
        arrays = fresh_arrays()
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
        suffix = f"d{args.startup_days}_s{args.startup_slots}"
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
                        "startup_days": definition["startup_days"],
                        "startup_slots": definition["startup_slots"],
                        "buy_start": start,
                        **core.metrics(
                            daily, start, protocol["known_stress_end"]
                        ),
                    }
                )
        pd.DataFrame(rows).to_csv(
            OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({"rows": len(rows), "grid": len(definitions)}, ensure_ascii=False))
        return

    rows = []
    full_rows = []
    frozen = []
    for definition in definitions:
        case_id = stable_id(definition)
        full_daily = run_case(
            arrays,
            score,
            order,
            definition,
            protocol,
            protocol["observation_end"],
        )
        full_metrics = core.metrics(
            full_daily, str(arrays["dates"][0]), protocol["observation_end"]
        )
        full_rows.append(
            {
                "case_id": case_id,
                "startup_days": definition["startup_days"],
                "startup_slots": definition["startup_slots"],
                **full_metrics,
            }
        )
        for anchor, starts in protocol["development_start_groups"].items():
            for start in starts:
                daily = run_case(
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
                        "startup_days": definition["startup_days"],
                        "startup_slots": definition["startup_slots"],
                        "anchor": anchor,
                        "buy_start": start,
                        **core.metrics(daily, start, protocol["observation_end"]),
                    }
                )
        frozen.append({"case_id": case_id, "definition": definition})
    pd.DataFrame(full_rows).to_csv(
        OUT / "observation_full_grid.csv", index=False, encoding="utf-8-sig"
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "development_start_results.csv", index=False, encoding="utf-8-sig"
    )
    summary = (
        frame.groupby(["startup_days", "startup_slots"])
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
