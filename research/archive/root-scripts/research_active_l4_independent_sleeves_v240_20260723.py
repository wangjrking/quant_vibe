# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_robust_objective_v56_20260721 as robust
import research_active_l4_score_trend_sizing_v234_20260723 as v234
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_independent_sleeves_v240_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_known_2026.json"
SLEEVE_NAMES = ("baseline", "trend_4d", "trend_5d", "trend_6d")


def stable_id(weights: dict) -> str:
    raw = json.dumps(weights, sort_keys=True, separators=(",", ":"))
    return "v240_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def aggregate_sleeves(
    sleeve_daily: dict[str, pd.DataFrame],
    total_initial_cash: float,
) -> pd.DataFrame:
    frames = []
    for name, daily in sleeve_daily.items():
        frame = daily[
            ["date", "equity", "turnover", "invested_ratio", "positions", "trades"]
        ].copy()
        frame = frame.rename(
            columns={
                "equity": f"equity_{name}",
                "turnover": f"turnover_{name}",
                "invested_ratio": f"invested_{name}",
                "positions": f"positions_{name}",
                "trades": f"trades_{name}",
            }
        )
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="inner", validate="one_to_one")
    equity_columns = [column for column in merged if column.startswith("equity_")]
    merged["equity"] = merged[equity_columns].sum(axis=1)
    previous_total = merged["equity"].shift(1).fillna(float(total_initial_cash))
    merged["return"] = merged["equity"] / previous_total - 1.0

    turnover_dollars = np.zeros(len(merged), dtype=float)
    invested_dollars = np.zeros(len(merged), dtype=float)
    positions = np.zeros(len(merged), dtype=float)
    trades = np.zeros(len(merged), dtype=float)
    for name in sleeve_daily:
        sleeve_equity = merged[f"equity_{name}"]
        sleeve_previous = sleeve_equity.shift(1).fillna(
            float(sleeve_daily[name].attrs["initial_cash"])
        )
        turnover_dollars += (
            merged[f"turnover_{name}"].to_numpy(dtype=float)
            * sleeve_previous.to_numpy(dtype=float)
        )
        invested_dollars += (
            merged[f"invested_{name}"].to_numpy(dtype=float)
            * sleeve_equity.to_numpy(dtype=float)
        )
        positions += merged[f"positions_{name}"].to_numpy(dtype=float)
        trades += merged[f"trades_{name}"].to_numpy(dtype=float)
    merged["turnover"] = turnover_dollars / previous_total.to_numpy(dtype=float)
    merged["invested_ratio"] = invested_dollars / merged["equity"].to_numpy(dtype=float)
    merged["positions"] = positions
    merged["trades"] = trades
    return merged[
        ["date", "return", "equity", "turnover", "invested_ratio", "positions", "trades"]
    ]


def run_weight_case(
    arrays: dict,
    score: np.ndarray,
    order: np.ndarray,
    weights: dict,
    protocol: dict,
    end: str,
    start: str | None = None,
) -> pd.DataFrame:
    sleeve_daily = {}
    base_definition = protocol["fixed_definition"]
    total_cash = float(protocol["total_initial_cash"])
    for name in SLEEVE_NAMES:
        weight = float(weights[name])
        if weight <= 0:
            continue
        sleeve_protocol = copy.deepcopy(protocol)
        sleeve_protocol["execution"]["initial_cash"] = total_cash * weight
        definition = {
            **base_definition,
            **protocol["sleeves"][name],
        }
        daily = v234.run_case(
            arrays,
            score,
            order,
            definition,
            sleeve_protocol,
            end,
            start,
        )
        daily.attrs["initial_cash"] = total_cash * weight
        sleeve_daily[name] = daily
    return aggregate_sleeves(sleeve_daily, total_cash)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent funded-sleeve portfolio research")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(full, 0.0, 7, 0.1)

    if args.open_known_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for item in frozen["candidates"]:
            for start in protocol["known_stress_buy_starts"]:
                daily = run_weight_case(
                    full,
                    score,
                    order,
                    item["weights"],
                    protocol,
                    protocol["known_stress_end"],
                    start,
                )
                rows.append(
                    {
                        "case_id": item["case_id"],
                        "weight_id": item["weight_id"],
                        "buy_start": start,
                        **core.metrics(daily, start, protocol["known_stress_end"]),
                    }
                )
        pd.DataFrame(rows).to_csv(
            OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps({"rows": len(rows)}, ensure_ascii=False))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    score_obs = score[: len(observation["dates"])]
    order_obs = order[: len(observation["dates"])]
    rows, definitions = [], {}
    for item in protocol["weight_grid"]:
        weights = {name: float(item[name]) for name in SLEEVE_NAMES}
        if abs(sum(weights.values()) - 1.0) > 1e-8:
            raise ValueError(f"weights do not sum to one: {item}")
        case_id = stable_id(weights)
        definitions[case_id] = {"weight_id": item["id"], "weights": weights}
        daily = run_weight_case(
            observation,
            score_obs,
            order_obs,
            weights,
            protocol,
            protocol["observation_end"],
        )
        rows.append(
            {
                "case_id": case_id,
                "weight_id": item["id"],
                **weights,
                **robust.evaluate_robust(daily, protocol),
            }
        )

    frame = pd.DataFrame(rows)
    gate = protocol["observation_gate"]
    frame["eligible"] = (
        (frame["min_year_cumulative_return"] >= gate["min_year_return"])
        & (frame["full_linear_annual_proxy"] >= gate["full_linear_annual_proxy"])
        & (frame["full_sharpe"] >= gate["full_sharpe"])
        & (frame["full_max_drawdown"] <= gate["max_drawdown"])
        & (frame["recent60_linear_annual_proxy"] >= gate["recent60_linear_annual_proxy"])
        & (frame["recent120_linear_annual_proxy"] >= gate["recent120_linear_annual_proxy"])
    )
    frame = frame.sort_values(
        ["eligible", "full_linear_annual_proxy", "min_year_cumulative_return", "full_sharpe"],
        ascending=[False, False, False, False],
    )
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [
        {
            "case_id": str(row.case_id),
            **definitions[str(row.case_id)],
            "observation_metrics": row.to_dict(),
        }
        for _, row in selected.iterrows()
    ]
    FROZEN.write_text(
        json.dumps(
            {
                "status": "frozen_before_known_2026_stress",
                "known_2026_used_for_this_grid_selection": False,
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "grid": len(frame),
                "eligible": int(frame["eligible"].sum()),
                "frozen": len(candidates),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
