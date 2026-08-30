# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_cold_start_trend_warmup_v252_20260723 as v252
import research_active_l4_trend_exposure_surface_v247_20260723 as v247
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_v252_start_path_v254_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    cases = {
        "v252_warmup50": (
            v252.run_case,
            v252.definition_for(protocol, 50),
        ),
        "v195_baseline": (
            v247.run_case,
            v247.definition_for(protocol, 1, 0.0, 1.0),
        ),
    }
    rows = []
    for strategy_id, (runner, definition) in cases.items():
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
    grouped = (
        frame.groupby(["strategy_id", "anchor"])
        .agg(
            starts=("buy_start", "count"),
            min_linear_annual_proxy=("linear_annual_proxy", "min"),
            median_linear_annual_proxy=("linear_annual_proxy", "median"),
            max_linear_annual_proxy=("linear_annual_proxy", "max"),
            min_sharpe=("sharpe", "min"),
            max_drawdown=("max_drawdown", "max"),
        )
        .reset_index()
    )
    grouped.to_csv(OUT / "start_path_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": len(frame), "overall_rows": len(overall)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
