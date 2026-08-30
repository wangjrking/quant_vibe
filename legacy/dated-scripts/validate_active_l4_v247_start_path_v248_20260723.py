# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_trend_exposure_surface_v247_20260723 as v247
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_v247_start_path_v248_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"


def definition(protocol: dict, strategy_id: str) -> dict:
    if strategy_id == "v247_l5_s1p0_e1p0":
        return v247.definition_for(protocol, 5, 1.0, 1.0)
    if strategy_id == "v195_baseline":
        return v247.definition_for(protocol, 1, 0.0, 1.0)
    raise ValueError(strategy_id)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    rows = []
    for strategy_id in protocol["strategies"]:
        params = definition(protocol, strategy_id)
        for anchor, starts in protocol["start_groups"].items():
            for start in starts:
                daily = v247.run_case(
                    arrays,
                    score,
                    order,
                    params,
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
    print(json.dumps({"rows": len(frame), "summary_rows": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
