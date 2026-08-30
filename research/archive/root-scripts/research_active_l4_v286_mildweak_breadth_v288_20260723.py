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
import research_active_l4_v286_weak_threshold_v287_20260723 as v287
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_v286_mildweak_breadth_v288_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN_GRID = OUT / "frozen_grid_before_known_2026.json"
SELECTED = OUT / "selected_candidate_before_known_2026.json"


def stable_id(definition):
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v288_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def definitions(protocol):
    base = v260.definition_for(protocol, protocol["position_warmup_days"])
    cases = [
        {**base, "entry_breadth_mode": mode, "entry_breadth_count_threshold": None}
        for mode in protocol["grid"]["baseline_modes"]
    ]
    for mode in protocol["grid"]["conditional_modes"]:
        for threshold in protocol["grid"]["breadth_count_threshold"]:
            cases.append(
                {
                    **base,
                    "entry_breadth_mode": mode,
                    "entry_breadth_count_threshold": int(threshold),
                }
            )
    return cases


def entry_slots(arrays, score, definition, protocol):
    momentum = v260.v258.v134.market_momentum(
        arrays, int(definition["market_lookback"])
    )
    breadth = np.sum(
        np.isfinite(score)
        & (score >= float(protocol["breadth_score_threshold"])),
        axis=1,
    )
    deep = float(protocol["deep_weak_momentum_threshold"])
    undefined = ~np.isfinite(momentum)
    negative = np.isfinite(momentum) & (momentum < 0.0)
    deep_weak = np.isfinite(momentum) & (momentum < deep)
    mild_weak = negative & ~deep_weak
    mode = definition["entry_breadth_mode"]
    trigger = undefined.copy()
    if mode == "all_negative":
        trigger |= negative
    elif mode == "deep_only":
        trigger |= deep_weak
    elif mode == "mild_low_breadth":
        trigger |= deep_weak | (
            mild_weak
            & (breadth < int(definition["entry_breadth_count_threshold"]))
        )
    elif mode == "mild_high_breadth":
        trigger |= deep_weak | (
            mild_weak
            & (breadth >= int(definition["entry_breadth_count_threshold"]))
        )
    else:
        raise ValueError(f"unknown entry breadth mode: {mode}")
    result = np.ones(len(momentum), dtype=np.int16)
    result[trigger] = 2
    return result


def run_case(arrays, score, order, definition, protocol, end, start=None, record_actions=False):
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
        entry_slots_schedule_override=entry_slots(
            arrays, score, definition, protocol
        ),
        split_candidate_target_override=True,
    )


def summarize_starts(frame):
    return (
        frame.groupby(["case_id", "entry_breadth_mode", "entry_breadth_count_threshold"], dropna=False)
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


def select_candidate(protocol, cases, full, starts, folds):
    by_id = {stable_id(item): item for item in cases}
    full_idx = full.set_index("case_id")
    starts_idx = summarize_starts(starts).set_index("case_id")
    fold_summary = folds.groupby("case_id").agg(
        fold_count=("fold_id", "count"),
        fold_min_cumulative_return=("cumulative_return", "min"),
        fold_min_linear_annual_proxy=("linear_annual_proxy", "min"),
        fold_median_linear_annual_proxy=("linear_annual_proxy", "median"),
        fold_min_sharpe=("sharpe", "min"),
        fold_max_drawdown=("max_drawdown", "max"),
    )
    baseline_id = next(
        case_id
        for case_id, definition in by_id.items()
        if definition["entry_breadth_mode"] == "all_negative"
    )
    baseline_full = float(full_idx.loc[baseline_id, "linear_annual_proxy"])
    baseline_fold_min = float(
        fold_summary.loc[baseline_id, "fold_min_linear_annual_proxy"]
    )
    rows = []
    for case_id, definition in by_id.items():
        f = full_idx.loc[case_id]
        s = starts_idx.loc[case_id]
        v = fold_summary.loc[case_id]
        eligible = bool(
            int(v["fold_count"]) == len(protocol["pseudo_validation_folds"])
            and float(v["fold_min_cumulative_return"]) > 0.0
            and float(f["linear_annual_proxy"]) >= baseline_full
            and float(v["fold_min_linear_annual_proxy"]) >= baseline_fold_min
            and float(f["sharpe"]) >= 1.25
            and float(f["max_drawdown"]) <= 0.30
            and float(s["min_linear_annual_proxy"]) >= 0.50
        )
        rows.append(
            {
                "case_id": case_id,
                "entry_breadth_mode": definition["entry_breadth_mode"],
                "entry_breadth_count_threshold": definition["entry_breadth_count_threshold"],
                "eligible": eligible,
                "full_linear_annual_proxy": f["linear_annual_proxy"],
                "full_sharpe": f["sharpe"],
                "full_max_drawdown": f["max_drawdown"],
                "fold_min_linear_annual_proxy": v["fold_min_linear_annual_proxy"],
                "fold_median_linear_annual_proxy": v["fold_median_linear_annual_proxy"],
                "starts_min_linear_annual_proxy": s["min_linear_annual_proxy"],
                "starts_median_linear_annual_proxy": s["median_linear_annual_proxy"],
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        [
            "eligible",
            "full_linear_annual_proxy",
            "fold_min_linear_annual_proxy",
            "starts_min_linear_annual_proxy",
        ],
        ascending=[False, False, False, False],
    )
    ranking.to_csv(OUT / "observation_selection_ranking.csv", index=False, encoding="utf-8-sig")
    valid = ranking[ranking["eligible"]]
    selected_id = str(valid.iloc[0]["case_id"]) if len(valid) else baseline_id
    payload = {
        "status": "selected_before_known_2026",
        "global_known_2026_exposure_before_design": True,
        "known_2026_used_for_selection": False,
        "stress_result_cannot_change_selected_candidate": True,
        "selected_case_id": selected_id,
        "selected_definition": by_id[selected_id],
        "fallback_case_id": baseline_id,
        "baseline_full_linear_annual_proxy": baseline_full,
        "baseline_fold_min_linear_annual_proxy": baseline_fold_min,
    }
    SELECTED.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_observation(protocol, arrays, score, order, cases):
    full_rows, start_rows, fold_rows = [], [], []
    for definition in cases:
        case_id = stable_id(definition)
        meta = {
            "case_id": case_id,
            "entry_breadth_mode": definition["entry_breadth_mode"],
            "entry_breadth_count_threshold": definition["entry_breadth_count_threshold"],
        }
        daily = run_case(arrays, score, order, definition, protocol, protocol["observation_end"])
        full_rows.append({**meta, **core.metrics(daily, str(arrays["dates"][0]), protocol["observation_end"])})
        for anchor, group in protocol["development_start_groups"].items():
            for start in group:
                result = run_case(arrays, score, order, definition, protocol, protocol["observation_end"], start)
                start_rows.append({**meta, "anchor": anchor, "buy_start": start, **core.metrics(result, start, protocol["observation_end"])})
        for fold in protocol["pseudo_validation_folds"]:
            result = run_case(arrays, score, order, definition, protocol, fold["end"], fold["start"])
            fold_rows.append({**meta, **fold, **core.metrics(result, fold["start"], fold["end"])})
    full = pd.DataFrame(full_rows)
    starts = pd.DataFrame(start_rows)
    folds = pd.DataFrame(fold_rows)
    full.to_csv(OUT / "observation_full_grid.csv", index=False, encoding="utf-8-sig")
    starts.to_csv(OUT / "development_start_results.csv", index=False, encoding="utf-8-sig")
    summarize_starts(starts).to_csv(OUT / "development_start_summary.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "pseudo_validation_folds.csv", index=False, encoding="utf-8-sig")
    FROZEN_GRID.write_text(
        json.dumps(
            {
                "status": "entire_grid_frozen_for_pre_2026_selection",
                "global_known_2026_exposure_before_design": True,
                "known_2026_used_for_grid_definition": False,
                "candidates": [{"case_id": stable_id(item), "definition": item} for item in cases],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    selected = select_candidate(protocol, cases, full, starts, folds)
    print(json.dumps({"grid": len(cases), "selected_case_id": selected["selected_case_id"]}, ensure_ascii=False))


def open_stress(protocol, arrays, score, order):
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    definition = selected["selected_definition"]
    rows = []
    for start in protocol["known_stress_starts"]:
        result = run_case(arrays, score, order, definition, protocol, protocol["known_stress_end"], start)
        rows.append(
            {
                "case_id": selected["selected_case_id"],
                "entry_breadth_mode": definition["entry_breadth_mode"],
                "entry_breadth_count_threshold": definition["entry_breadth_count_threshold"],
                "buy_start": start,
                "independent_validation": False,
                "can_change_selected_candidate": False,
                **core.metrics(result, start, protocol["known_stress_end"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "known_2026_stress_selected_only.csv", index=False, encoding="utf-8-sig")
    summary = {
        "case_id": selected["selected_case_id"],
        "entry_breadth_mode": definition["entry_breadth_mode"],
        "entry_breadth_count_threshold": definition["entry_breadth_count_threshold"],
        "independent_validation": False,
        "can_change_selected_candidate": False,
        "min_linear_annual_proxy": float(frame["linear_annual_proxy"].min()),
        "median_linear_annual_proxy": float(frame["linear_annual_proxy"].median()),
        "min_sharpe": float(frame["sharpe"].min()),
        "max_drawdown": float(frame["max_drawdown"].max()),
    }
    (OUT / "known_2026_stress_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def export_actions(protocol, arrays, score, order):
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
    daily.to_csv(OUT / "daily_selected_observation.csv", index=False, encoding="utf-8-sig")
    actions.to_csv(OUT / "replay_actions_selected_observation.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"case_id": selected["selected_case_id"], "daily_rows": len(daily), "action_rows": len(actions)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="V288轻弱势市场宽度分散研究")
    parser.add_argument("--open-known-2026", action="store_true")
    parser.add_argument("--export-selected-observation-actions", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    if args.export_selected_observation_actions:
        export_actions(protocol, arrays, score, order)
    elif args.open_known_2026:
        open_stress(protocol, arrays, score, order)
    else:
        run_observation(protocol, arrays, score, order, definitions(protocol))


if __name__ == "__main__":
    main()
