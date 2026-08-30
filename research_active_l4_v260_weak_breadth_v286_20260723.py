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
    / "quant/data_file/reports/strategy_agent_active_l4_v260_weak_breadth_v286_20260723"
)
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN_GRID = OUT / "frozen_grid_before_known_2026.json"
SELECTED = OUT / "selected_candidate_before_known_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v286_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def definitions(protocol: dict) -> list[dict]:
    base = v260.definition_for(protocol, protocol["position_warmup_days"])
    return [
        {**base, "weak_daily_entry_slots": int(value)}
        for value in protocol["grid"]["weak_daily_entry_slots"]
    ]


def entry_slots_schedule(arrays, definition, protocol):
    momentum = v260.v258.v134.market_momentum(
        arrays, int(definition["market_lookback"])
    )
    slots = np.full(
        len(momentum),
        int(protocol["grid"]["normal_strong_daily_entry_slots"]),
        dtype=np.int16,
    )
    weak = ~np.isfinite(momentum) | (
        momentum < float(definition["market_return_min"])
    )
    slots[weak] = int(definition["weak_daily_entry_slots"])
    return slots


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
    v252 = v260.v258.v252
    return v252.v162.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start,
        record_actions=record_actions,
        selection_mask_override=v252.v174.selection_mask(
            arrays, definition["max_rank_deterioration"]
        ),
        candidate_target_multiplier_override=v252.warmup_multiplier(
            arrays, score, definition, protocol, start
        ),
        max_positions_override=int(definition["max_positions"]),
        max_positions_schedule_override=v260.position_schedule(
            arrays, definition, start
        ),
        entry_slots_schedule_override=entry_slots_schedule(
            arrays, definition, protocol
        ),
        split_candidate_target_override=True,
    )


def start_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["case_id", "weak_daily_entry_slots"])
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


def choose(protocol, cases, full, starts, folds):
    full_idx = full.set_index("case_id")
    starts_idx = start_summary(starts).set_index("case_id")
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
    by_id = {stable_id(item): item for item in cases}
    baseline_id = next(
        case_id
        for case_id, definition in by_id.items()
        if definition["weak_daily_entry_slots"]
        == protocol["selection_rule"]["baseline_weak_daily_entry_slots"]
    )
    rows = []
    for case_id, definition in by_id.items():
        f = full_idx.loc[case_id]
        s = starts_idx.loc[case_id]
        v = fold_summary.loc[case_id]
        eligible = bool(
            int(v["fold_count"]) == len(protocol["pseudo_validation_folds"])
            and float(v["fold_min_cumulative_return"]) > 0.0
            and float(f["linear_annual_proxy"]) >= 0.40
            and float(f["sharpe"]) >= 1.0
            and float(f["max_drawdown"]) <= 0.30
            and float(s["min_linear_annual_proxy"]) >= 0.35
        )
        rows.append(
            {
                "case_id": case_id,
                "weak_daily_entry_slots": definition[
                    "weak_daily_entry_slots"
                ],
                "eligible": eligible,
                "fold_min_linear_annual_proxy": v[
                    "fold_min_linear_annual_proxy"
                ],
                "fold_median_linear_annual_proxy": v[
                    "fold_median_linear_annual_proxy"
                ],
                "fold_min_sharpe": v["fold_min_sharpe"],
                "fold_max_drawdown": v["fold_max_drawdown"],
                "starts_min_linear_annual_proxy": s[
                    "min_linear_annual_proxy"
                ],
                "starts_median_linear_annual_proxy": s[
                    "median_linear_annual_proxy"
                ],
                "full_linear_annual_proxy": f["linear_annual_proxy"],
                "full_sharpe": f["sharpe"],
                "full_max_drawdown": f["max_drawdown"],
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
    valid = ranking[ranking["eligible"]]
    selected_id = str(valid.iloc[0]["case_id"]) if len(valid) else baseline_id
    payload = {
        "status": "selected_before_known_2026",
        "global_known_2026_exposure_before_design": True,
        "known_2026_used_for_selection": False,
        "stress_result_cannot_change_selected_candidate": True,
        "selected_case_id": selected_id,
        "selected_definition": by_id[selected_id],
        "baseline_case_id": baseline_id,
        "selection_ranking_file": "observation_selection_ranking.csv",
    }
    SELECTED.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def run_observation(protocol, arrays, score, order, cases):
    full_rows, start_rows, fold_rows = [], [], []
    for definition in cases:
        case_id = stable_id(definition)
        slots = definition["weak_daily_entry_slots"]
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
                "weak_daily_entry_slots": slots,
                **core.metrics(
                    daily,
                    str(arrays["dates"][0]),
                    protocol["observation_end"],
                ),
            }
        )
        for anchor, group in protocol["development_start_groups"].items():
            for start in group:
                result = run_case(
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
                        "weak_daily_entry_slots": slots,
                        "anchor": anchor,
                        "buy_start": start,
                        **core.metrics(
                            result, start, protocol["observation_end"]
                        ),
                    }
                )
        for fold in protocol["pseudo_validation_folds"]:
            result = run_case(
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
                    "weak_daily_entry_slots": slots,
                    **fold,
                    **core.metrics(result, fold["start"], fold["end"]),
                }
            )
    full = pd.DataFrame(full_rows)
    starts = pd.DataFrame(start_rows)
    folds = pd.DataFrame(fold_rows)
    full.to_csv(
        OUT / "observation_full_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    starts.to_csv(
        OUT / "development_start_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    start_summary(starts).to_csv(
        OUT / "development_start_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    folds.to_csv(
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
    selected = choose(protocol, cases, full, starts, folds)
    print(
        json.dumps(
            {
                "grid": len(cases),
                "start_rows": len(starts),
                "fold_rows": len(folds),
                "selected_case_id": selected["selected_case_id"],
            },
            ensure_ascii=False,
        )
    )


def open_stress(protocol, arrays, score, order):
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    definition = selected["selected_definition"]
    rows = []
    for start in protocol["known_stress_starts"]:
        result = run_case(
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
                "weak_daily_entry_slots": definition[
                    "weak_daily_entry_slots"
                ],
                "buy_start": start,
                "independent_validation": False,
                "can_change_selected_candidate": False,
                **core.metrics(result, start, protocol["known_stress_end"]),
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
        "weak_daily_entry_slots": definition["weak_daily_entry_slots"],
        "independent_validation": False,
        "can_change_selected_candidate": False,
        "min_linear_annual_proxy": float(frame["linear_annual_proxy"].min()),
        "median_linear_annual_proxy": float(
            frame["linear_annual_proxy"].median()
        ),
        "mean_linear_annual_proxy": float(frame["linear_annual_proxy"].mean()),
        "min_sharpe": float(frame["sharpe"].min()),
        "max_drawdown": float(frame["max_drawdown"].max()),
    }
    (OUT / "known_2026_stress_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="V260 弱势市场每日新增分散研究")
    parser.add_argument("--open-known-2026", action="store_true")
    parser.add_argument("--export-selected-observation-actions", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    cases = definitions(protocol)
    if args.export_selected_observation_actions:
        selected = json.loads(SELECTED.read_text(encoding="utf-8"))
        daily, actions = run_case(
            arrays,
            score,
            order,
            selected["selected_definition"],
            protocol,
            protocol["observation_end"],
            record_actions=True,
        )
        daily_path = OUT / "daily_selected_observation.csv"
        actions_path = OUT / "replay_actions_selected_observation.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        actions.to_csv(actions_path, index=False, encoding="utf-8-sig")
        print(
            json.dumps(
                {
                    "case_id": selected["selected_case_id"],
                    "daily_rows": len(daily),
                    "action_rows": len(actions),
                    "daily_path": str(daily_path),
                    "actions_path": str(actions_path),
                },
                ensure_ascii=False,
            )
        )
    elif args.open_known_2026:
        open_stress(protocol, arrays, score, order)
    else:
        run_observation(protocol, arrays, score, order, cases)


if __name__ == "__main__":
    main()
