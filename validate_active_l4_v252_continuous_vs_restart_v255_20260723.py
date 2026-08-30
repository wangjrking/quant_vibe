# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_cold_start_trend_warmup_v252_20260723 as v252
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_v252_continuous_vs_restart_v255_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    definition = v252.definition_for(protocol, 50)
    continuous_daily = v252.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        protocol["end"],
    )
    restart = pd.read_csv(ROOT / protocol["restart_results"])
    restart = restart[restart["strategy_id"] == "v252_warmup50"].copy()
    rows = []
    for row in restart.itertuples(index=False):
        continuous = core.metrics(
            continuous_daily, str(row.buy_start), protocol["end"]
        )
        rows.append(
            {
                "anchor": row.anchor,
                "buy_start": str(row.buy_start),
                "restart_linear_annual_proxy": float(row.linear_annual_proxy),
                "continuous_linear_annual_proxy": continuous[
                    "linear_annual_proxy"
                ],
                "annual_proxy_gap_restart_minus_continuous": float(
                    row.linear_annual_proxy
                    - continuous["linear_annual_proxy"]
                ),
                "restart_sharpe": float(row.sharpe),
                "continuous_sharpe": continuous["sharpe"],
                "restart_max_drawdown": float(row.max_drawdown),
                "continuous_max_drawdown": continuous["max_drawdown"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "continuous_vs_restart.csv", index=False, encoding="utf-8-sig")
    summary = {
        "starts": int(len(frame)),
        "restart_min_linear_annual_proxy": float(
            frame["restart_linear_annual_proxy"].min()
        ),
        "continuous_min_linear_annual_proxy": float(
            frame["continuous_linear_annual_proxy"].min()
        ),
        "restart_median_linear_annual_proxy": float(
            frame["restart_linear_annual_proxy"].median()
        ),
        "continuous_median_linear_annual_proxy": float(
            frame["continuous_linear_annual_proxy"].median()
        ),
        "median_abs_annual_proxy_gap": float(
            frame["annual_proxy_gap_restart_minus_continuous"].abs().median()
        ),
        "max_abs_annual_proxy_gap": float(
            frame["annual_proxy_gap_restart_minus_continuous"].abs().max()
        ),
        "restart_better_count": int(
            (frame["annual_proxy_gap_restart_minus_continuous"] > 0).sum()
        ),
        "continuous_better_count": int(
            (frame["annual_proxy_gap_restart_minus_continuous"] < 0).sum()
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
