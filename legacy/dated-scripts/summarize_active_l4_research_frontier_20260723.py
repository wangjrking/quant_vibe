# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "quant/data_file/reports"
OUT = REPORT_ROOT / "strategy_agent_active_l4_research_frontier_20260723"
EXPECTED_CACHE = (
    "quant/data_file/reports/strategy_agent_active_l4_liquidity_risk_v59_20260721/"
    "active_formal_rank_liquidity_arrays_v2.npz"
)


def comparable(protocol: dict) -> bool:
    execution = protocol.get("execution", {})
    return (
        str(protocol.get("observation_end")) == "20251231"
        and str(protocol.get("known_stress_end")) == "20260720"
        and str(protocol.get("input_cache")) == EXPECTED_CACHE
        and abs(float(execution.get("fixed_slippage_ratio", -1)) - 0.003) < 1e-12
        and int(execution.get("initial_cash", -1)) == 700000
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    observation_rows = []
    stress_rows = []
    inventory = []
    for directory in sorted(REPORT_ROOT.glob("strategy_agent_active_l4_*")):
        protocol_path = directory / "preregistered_protocol.json"
        observation_path = directory / "observation_grid.csv"
        stress_path = directory / "known_2026_stress.csv"
        if not protocol_path.exists() or not observation_path.exists() or not stress_path.exists():
            continue
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not comparable(protocol):
            continue
        observation = pd.read_csv(observation_path)
        stress = pd.read_csv(stress_path)
        required_observation = {
            "case_id",
            "full_linear_annual_proxy",
            "full_sharpe",
            "full_max_drawdown",
            "min_year_cumulative_return",
        }
        required_stress = {"case_id", "linear_annual_proxy", "sharpe", "max_drawdown"}
        if not required_observation.issubset(observation.columns) or not required_stress.issubset(stress.columns):
            continue
        if "period" in stress.columns:
            preferred = stress[stress["period"].astype(str) == "through_20260720"]
            if not preferred.empty:
                stress = preferred
        observation = observation.copy()
        stress = stress.copy()
        observation["source_report"] = directory.name
        stress["source_report"] = directory.name
        observation_rows.append(observation)
        stress_rows.append(stress)
        inventory.append(
            {
                "source_report": directory.name,
                "observation_rows": len(observation),
                "stress_rows": len(stress),
                "stress_candidates": stress["case_id"].nunique(),
            }
        )

    if not observation_rows or not stress_rows:
        raise RuntimeError("没有找到同口径观察期和2026压力结果")
    observations = pd.concat(observation_rows, ignore_index=True, sort=False)
    stresses = pd.concat(stress_rows, ignore_index=True, sort=False)
    stress_summary = (
        stresses.groupby(["source_report", "case_id"], as_index=False)
        .agg(
            stress_start_count=("linear_annual_proxy", "size"),
            stress_min_annual=("linear_annual_proxy", "min"),
            stress_median_annual=("linear_annual_proxy", "median"),
            stress_max_annual=("linear_annual_proxy", "max"),
            stress_min_sharpe=("sharpe", "min"),
            stress_max_drawdown=("max_drawdown", "max"),
        )
    )
    merged = observations.merge(stress_summary, on=["source_report", "case_id"], how="inner")
    merged = merged[merged["stress_start_count"] >= 5].copy()
    baseline_annual = 1.025102417950971
    baseline_stress_min = 0.955855
    baseline_stress_median = 1.185235
    merged["beats_baseline_observation"] = merged["full_linear_annual_proxy"] > baseline_annual
    merged["beats_baseline_stress_min"] = merged["stress_min_annual"] > baseline_stress_min
    merged["beats_baseline_stress_median"] = merged["stress_median_annual"] > baseline_stress_median
    merged["dual_improvement"] = (
        merged["beats_baseline_observation"]
        & merged["beats_baseline_stress_min"]
        & merged["beats_baseline_stress_median"]
    )
    merged["joint_floor"] = np.minimum(
        merged["full_linear_annual_proxy"], merged["stress_min_annual"]
    )
    merged = merged.sort_values(
        [
            "dual_improvement",
            "joint_floor",
            "full_linear_annual_proxy",
            "stress_median_annual",
            "full_sharpe",
        ],
        ascending=[False, False, False, False, False],
    )
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(OUT / "comparable_report_inventory.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(OUT / "joint_frontier_all.csv", index=False, encoding="utf-8-sig")
    merged.head(100).to_csv(OUT / "joint_frontier_top100.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "research_only_read_only_summary",
        "comparable_reports": int(inventory_frame["source_report"].nunique()),
        "comparable_joined_candidates": int(len(merged)),
        "dual_improvement_candidates": int(merged["dual_improvement"].sum()),
        "baseline": {
            "observation_linear_annual_proxy": baseline_annual,
            "stress_min_linear_annual_proxy": baseline_stress_min,
            "stress_median_linear_annual_proxy": baseline_stress_median,
        },
        "top_rows": merged.head(20)[
            [
                "source_report",
                "case_id",
                "full_linear_annual_proxy",
                "full_sharpe",
                "full_max_drawdown",
                "stress_min_annual",
                "stress_median_annual",
                "stress_min_sharpe",
                "stress_max_drawdown",
                "dual_improvement",
            ]
        ].to_dict("records"),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
