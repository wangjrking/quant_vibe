# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
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
    / "quant/data_file/reports/strategy_agent_active_l4_v260_age15_exit_v285_20260723"
)
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN_GRID = OUT / "frozen_grid_before_known_2026.json"
SELECTED = OUT / "selected_candidate_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v285_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def definitions(protocol: dict) -> list[dict]:
    base = v260.definition_for(protocol, protocol["position_warmup_days"])
    age = int(protocol["grid"]["stale_age_days"])
    return [
        {
            **base,
            "replacement_advantage_age_bands": [
                [0, float(base["replacement_advantage"])],
                [age, float(value)],
            ],
            "stale_replacement_advantage": float(value),
        }
        for value in protocol["grid"]["stale_replacement_advantage"]
    ]


def run_case(arrays, score, order, definition, protocol, end, start=None):
    v252 = v260.v258.v252
    return v252.v162.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start,
        selection_mask_override=v252.v174.selection_mask(
            arrays, definition["max_rank_deterioration"]
        ),
        candidate_target_multiplier_override=v252.warmup_multiplier(
            arrays, score, definition, protocol, start
        ),
        replacement_advantage_age_bands_override=definition[
            "replacement_advantage_age_bands"
        ],
        max_positions_override=int(definition["max_positions"]),
        max_positions_schedule_override=v260.position_schedule(
            arrays, definition, start
        ),
    )


def summarize_starts(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["case_id", "stale_replacement_advantage"])
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


def select_candidate(
    protocol: dict,
    cases: list[dict],
    full: pd.DataFrame,
    starts: pd.DataFrame,
    folds: pd.DataFrame,
) -> dict:
    value_by_id = {
        stable_id(definition): float(definition["stale_replacement_advantage"])
        for definition in cases
    }
    baseline_value = float(
        protocol["selection_rule"]["baseline_stale_replacement_advantage"]
    )
    baseline_id = next(
        case_id
        for case_id, value in value_by_id.items()
        if np.isclose(value, baseline_value)
    )
    full_idx = full.set_index("case_id")
    start_summary = summarize_starts(starts).set_index("case_id")
    fold_summary = (
        folds.groupby("case_id")
        .agg(
            fold_count=("fold_id", "count"),
            fold_min_cumulative_return=("cumulative_return", "min"),
            fold_min_linear_annual_proxy=("linear_annual_proxy", "min"),
            fold_median_linear_annual_proxy=("linear_annual_proxy", "median"),
            fold_min_sharpe=("sharpe", "min"),
            fold_max_drawdown=("max_drawdown", "max"),
        )
    )
    baseline_full = full_idx.loc[baseline_id]
    baseline_start = start_summary.loc[baseline_id]

    rows = []
    definitions_by_id = {stable_id(item): item for item in cases}
    for case_id, definition in definitions_by_id.items():
        full_row = full_idx.loc[case_id]
        start_row = start_summary.loc[case_id]
        fold_row = fold_summary.loc[case_id]
        eligible = bool(
            int(fold_row["fold_count"])
            == len(protocol["pseudo_validation_folds"])
            and float(fold_row["fold_min_cumulative_return"]) > 0.0
            and float(full_row["linear_annual_proxy"])
            >= 0.90 * float(baseline_full["linear_annual_proxy"])
            and float(full_row["sharpe"])
            >= float(baseline_full["sharpe"]) - 0.05
            and float(full_row["max_drawdown"])
            <= float(baseline_full["max_drawdown"]) + 0.03
            and float(start_row["min_linear_annual_proxy"])
            >= 0.95 * float(baseline_start["min_linear_annual_proxy"])
        )
        rows.append(
            {
                "case_id": case_id,
                "stale_replacement_advantage": definition[
                    "stale_replacement_advantage"
                ],
                "eligible": eligible,
                **{
                    f"full_{key}": full_row[key]
                    for key in [
                        "linear_annual_proxy",
                        "sharpe",
                        "max_drawdown",
                        "cumulative_return",
                    ]
                },
                **{
                    f"starts_{key}": start_row[key]
                    for key in [
                        "min_linear_annual_proxy",
                        "median_linear_annual_proxy",
                        "mean_linear_annual_proxy",
                        "min_sharpe",
                        "max_drawdown",
                    ]
                },
                **fold_row.to_dict(),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        [
            "eligible",
            "fold_min_linear_annual_proxy",
            "starts_min_linear_annual_proxy",
            "full_linear_annual_proxy",
        ],
        ascending=[False, False, False, False],
    )
    ranking.to_csv(
        OUT / "observation_selection_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    eligible = ranking[ranking["eligible"]]
    selected_id = (
        str(eligible.iloc[0]["case_id"]) if not eligible.empty else baseline_id
    )
    return {
        "status": "selected_before_known_2026",
        "global_known_2026_exposure_before_design": True,
        "known_2026_used_for_selection": False,
        "stress_result_cannot_change_selected_candidate": True,
        "selected_case_id": selected_id,
        "selected_definition": definitions_by_id[selected_id],
        "baseline_case_id": baseline_id,
        "selection_ranking_file": "observation_selection_ranking.csv",
    }


def run_observation(protocol: dict, arrays: dict, score, order, cases):
    full_rows = []
    start_rows = []
    fold_rows = []
    for definition in cases:
        case_id = stable_id(definition)
        value = definition["stale_replacement_advantage"]
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
                "stale_replacement_advantage": value,
                **core.metrics(
                    daily,
                    str(arrays["dates"][0]),
                    protocol["observation_end"],
                ),
            }
        )
        for anchor, group in protocol["development_start_groups"].items():
            for start in group:
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
                        "stale_replacement_advantage": value,
                        "anchor": anchor,
                        "buy_start": start,
                        **core.metrics(
                            start_daily, start, protocol["observation_end"]
                        ),
                    }
                )
        for fold in protocol["pseudo_validation_folds"]:
            fold_daily = run_case(
                arrays,
                score,
                order,
                definition,
                protocol,
                fold["end"],
                fold["start"],
            )
            fold_rows.append(
                {
                    "case_id": case_id,
                    "stale_replacement_advantage": value,
                    "fold_id": fold["fold_id"],
                    "start": fold["start"],
                    "end": fold["end"],
                    **core.metrics(fold_daily, fold["start"], fold["end"]),
                }
            )

    full_frame = pd.DataFrame(full_rows)
    start_frame = pd.DataFrame(start_rows)
    fold_frame = pd.DataFrame(fold_rows)
    full_frame.to_csv(
        OUT / "observation_full_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    start_frame.to_csv(
        OUT / "development_start_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize_starts(start_frame).to_csv(
        OUT / "development_start_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_frame.to_csv(
        OUT / "pseudo_validation_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    FROZEN_GRID.write_text(
        json.dumps(
            {
                "status": "entire_grid_frozen_for_pre_2026_selection",
                "global_known_2026_exposure_before_design": True,
                "known_2026_used_for_grid_definition": False,
                "candidates": [
                    {"case_id": stable_id(item), "definition": item}
                    for item in cases
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    selected = select_candidate(
        protocol, cases, full_frame, start_frame, fold_frame
    )
    SELECTED.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "grid": len(cases),
                "development_start_rows": len(start_frame),
                "pseudo_validation_rows": len(fold_frame),
                "selected_case_id": selected["selected_case_id"],
            },
            ensure_ascii=False,
        )
    )


def open_known_stress(protocol: dict, arrays: dict, score, order):
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    if selected["status"] != "selected_before_known_2026":
        raise RuntimeError("候选未在2025年底以前的口径完成冻结")
    definition = selected["selected_definition"]
    rows = []
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
                "case_id": selected["selected_case_id"],
                "stale_replacement_advantage": definition[
                    "stale_replacement_advantage"
                ],
                "buy_start": start,
                "independent_validation": False,
                "can_change_selected_candidate": False,
                **core.metrics(daily, start, protocol["known_stress_end"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "known_2026_stress_selected_only.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "case_id": selected["selected_case_id"],
        "independent_validation": False,
        "can_change_selected_candidate": False,
        "starts": len(frame),
        "min_linear_annual_proxy": float(frame["linear_annual_proxy"].min()),
        "median_linear_annual_proxy": float(
            frame["linear_annual_proxy"].median()
        ),
        "mean_linear_annual_proxy": float(frame["linear_annual_proxy"].mean()),
        "min_sharpe": float(frame["sharpe"].min()),
        "max_drawdown": float(frame["max_drawdown"].max()),
    }
    (OUT / "known_2026_stress_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 持仓满15日后替换门槛研究")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    cases = definitions(protocol)
    if args.open_known_2026:
        open_known_stress(protocol, arrays, score, order)
    else:
        run_observation(protocol, arrays, score, order, cases)


if __name__ == "__main__":
    main()
