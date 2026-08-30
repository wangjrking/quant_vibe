# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_regime_trend_window_v250_20260723 as v250
import research_active_l4_trend_exposure_surface_v247_20260723 as v247
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_v250_start_path_v251_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    definitions = {
        "v250_switch_0p02": v250.definition_for(protocol, 0.02),
        "v195_baseline": v247.definition_for(protocol, 1, 0.0, 1.0),
    }
    rows = []
    for strategy_id, definition in definitions.items():
        runner = v250.run_case if strategy_id.startswith("v250") else v247.run_case
        for anchor, starts in protocol["start_groups"].items():
            for start in starts:
                daily = runner(
                    arrays,
                    score,
                    order,
                    definition,
                    protocol,
                    protocol["end"],
                    start,
                )
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "anchor": anchor,
                        "buy_start": start,
                        **core.metrics(daily, start, protocol["end"]),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "start_path_results.csv", index=False, encoding="utf-8-sig")
    summary = (
        frame.groupby(["strategy_id", "anchor"])
        .agg(
            starts=("buy_start", "count"),
            min_linear_annual_proxy=("linear_annual_proxy", "min"),
            median_linear_annual_proxy=("linear_annual_proxy", "median"),
            max_linear_annual_proxy=("linear_annual_proxy", "max"),
            min_sharpe=("sharpe", "min"),
            median_sharpe=("sharpe", "median"),
            max_drawdown=("max_drawdown", "max"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "start_path_summary.csv", index=False, encoding="utf-8-sig")
    overall = (
        frame.groupby("strategy_id")
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
    overall.to_csv(OUT / "start_path_overall.csv", index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "rows": len(frame),
                "summary_rows": len(summary),
                "overall_rows": len(overall),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
