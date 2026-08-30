# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_cold_start_trend_warmup_v252_20260723 as v252
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "quant/data_file/reports/strategy_agent_active_l4_v260_paths_v261_20260723"
)
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
        "v260_regime_position_warmup50": (
            v260.run_case,
            v260.definition_for(protocol, 50),
        ),
    }
    continuous = {
        strategy_id: runner(
            arrays,
            score,
            order,
            definition,
            protocol,
            protocol["end"],
        )
        for strategy_id, (runner, definition) in cases.items()
    }
    rows = []
    for strategy_id, (runner, definition) in cases.items():
        for anchor, starts in protocol["start_groups"].items():
            for start in starts:
                restart_daily = runner(
                    arrays,
                    score,
                    order,
                    definition,
                    protocol,
                    protocol["end"],
                    start,
                )
                restart = core.metrics(
                    restart_daily, start, protocol["end"]
                )
                sliced = core.metrics(
                    continuous[strategy_id], start, protocol["end"]
                )
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "anchor": anchor,
                        "buy_start": start,
                        **{
                            f"restart_{key}": value
                            for key, value in restart.items()
                        },
                        **{
                            f"continuous_{key}": value
                            for key, value in sliced.items()
                        },
                        "annual_gap_restart_minus_continuous": (
                            restart["linear_annual_proxy"]
                            - sliced["linear_annual_proxy"]
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "path_results.csv", index=False, encoding="utf-8-sig"
    )
    (
        frame.groupby("strategy_id")
        .agg(
            starts=("buy_start", "count"),
            restart_min_annual=("restart_linear_annual_proxy", "min"),
            restart_median_annual=(
                "restart_linear_annual_proxy",
                "median",
            ),
            restart_mean_annual=("restart_linear_annual_proxy", "mean"),
            restart_min_sharpe=("restart_sharpe", "min"),
            restart_max_drawdown=("restart_max_drawdown", "max"),
            continuous_min_annual=(
                "continuous_linear_annual_proxy",
                "min",
            ),
            continuous_median_annual=(
                "continuous_linear_annual_proxy",
                "median",
            ),
            median_abs_annual_gap=(
                "annual_gap_restart_minus_continuous",
                lambda values: values.abs().median(),
            ),
            max_abs_annual_gap=(
                "annual_gap_restart_minus_continuous",
                lambda values: values.abs().max(),
            ),
        )
        .reset_index()
        .to_csv(
            OUT / "path_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )
    print(json.dumps({"rows": len(frame)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
