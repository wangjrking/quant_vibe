# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_cold_start_trend_warmup_v252_20260723 as v252
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_cold_start_warmup_neighborhood_v253_20260723"
PROTOCOL = OUT / "preregistered_protocol.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Active L4 冷启动预热天数邻域验证")
    parser.add_argument("--open-known-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    groups = (
        {"2026压力": protocol["known_stress_starts"]}
        if args.open_known_2026
        else protocol["development_start_groups"]
    )
    end = protocol["known_stress_end"] if args.open_known_2026 else protocol["observation_end"]
    rows = []
    full_rows = []
    for days in protocol["grid"]["warmup_days"]:
        definition = v252.definition_for(protocol, int(days))
        if not args.open_known_2026:
            full_daily = v252.run_case(
                arrays,
                score,
                order,
                definition,
                protocol,
                end,
            )
            full_rows.append(
                {
                    "warmup_days": int(days),
                    **core.metrics(full_daily, str(arrays["dates"][0]), end),
                }
            )
        for anchor, starts in groups.items():
            for start in starts:
                daily = v252.run_case(
                    arrays,
                    score,
                    order,
                    definition,
                    protocol,
                    end,
                    start,
                )
                rows.append(
                    {
                        "warmup_days": int(days),
                        "anchor": anchor,
                        "buy_start": start,
                        **core.metrics(daily, start, end),
                    }
                )
    frame = pd.DataFrame(rows)
    prefix = "known_2026" if args.open_known_2026 else "development"
    frame.to_csv(OUT / f"{prefix}_start_results.csv", index=False, encoding="utf-8-sig")
    if full_rows:
        pd.DataFrame(full_rows).to_csv(
            OUT / "observation_full_metrics.csv", index=False, encoding="utf-8-sig"
        )
    summary = (
        frame.groupby("warmup_days")
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
    summary.to_csv(OUT / f"{prefix}_start_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": len(frame), "summary_rows": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
