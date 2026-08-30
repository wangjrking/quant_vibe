# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "quant/data_file/reports/strategy_agent_active_l4_liquidity_risk_v59_20260721/active_formal_rank_liquidity_arrays_v2.npz"
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_filter_ablation_v85_20260722"


def net_return(buy: np.ndarray, sell: np.ndarray, slippage: float) -> np.ndarray:
    buy_cost = buy * (1.0 + slippage) * (1.0 + 0.0003)
    sell_value = sell * (1.0 - slippage) * (1.0 - 0.0003 - 0.0005)
    return sell_value / buy_cost - 1.0


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"days": 0, "mean_return": float("nan"), "positive_ratio": float("nan")}
    return {
        "days": int(len(values)),
        "mean_return": float(np.mean(values)),
        "positive_ratio": float(np.mean(values > 0.0)),
    }


def main() -> None:
    with np.load(CACHE, allow_pickle=False) as saved:
        a = {key: saved[key] for key in saved.files}

    dates = a["dates"].astype(str)
    stocks = a["stocks"].astype(str)
    score = a["rank_10d"]
    board_rate = core.board_limit_rate(stocks)
    future_sell = np.full_like(a["buy_open"], np.nan, dtype=np.float32)
    future_sell[:-11] = a["buy_open"][11:]

    valid = (
        np.isfinite(score)
        & np.isfinite(a["buy_open"])
        & (a["buy_open"] > 0)
        & np.isfinite(a["buy_pre_close"])
        & (a["buy_pre_close"] > 0)
        & np.isfinite(future_sell)
        & (future_sell > 0)
    )
    stages = [
        ("S0_raw_score_and_price", valid),
        ("S1_signal_and_buy_clean", valid & a["signal_clean"] & a["buy_clean"]),
        ("S2_listed_60d", valid & a["signal_clean"] & a["buy_clean"] & (a["listed_days"] >= 60)),
        (
            "S3_amount_9w",
            valid
            & a["signal_clean"]
            & a["buy_clean"]
            & (a["listed_days"] >= 60)
            & np.isfinite(a["amount"])
            & (a["amount"] >= 90000),
        ),
        (
            "S4_total_mv_20b",
            valid
            & a["signal_clean"]
            & a["buy_clean"]
            & (a["listed_days"] >= 60)
            & np.isfinite(a["amount"])
            & (a["amount"] >= 90000)
            & np.isfinite(a["total_mv"])
            & (a["total_mv"] >= 200000),
        ),
        (
            "S5_turnover_le_10",
            valid
            & a["signal_clean"]
            & a["buy_clean"]
            & (a["listed_days"] >= 60)
            & np.isfinite(a["amount"])
            & (a["amount"] >= 90000)
            & np.isfinite(a["total_mv"])
            & (a["total_mv"] >= 200000)
            & np.isfinite(a["turnover_rate"])
            & (a["turnover_rate"] >= 0)
            & (a["turnover_rate"] <= 10),
        ),
    ]
    limit_up = a["buy_open"] >= a["buy_pre_close"] * (1.0 + board_rate[None, :]) * 0.995
    stages.append(("S6_not_limit_up", stages[-1][1] & ~limit_up))

    rows = []
    overlap_rows = []
    for t, signal_date in enumerate(dates):
        raw_idx = np.flatnonzero(stages[0][1][t])
        if len(raw_idx) == 0:
            continue
        raw_top = raw_idx[np.argsort(-score[t, raw_idx], kind="stable")[:3]]
        for stage_name, mask in stages:
            idx = np.flatnonzero(mask[t])
            if len(idx) == 0:
                continue
            selected = idx[np.argsort(-score[t, idx], kind="stable")[:3]]
            gross = future_sell[t, selected] / a["buy_open"][t, selected] - 1.0
            for slippage in (0.001, 0.003):
                net = net_return(a["buy_open"][t, selected], future_sell[t, selected], slippage)
                rows.append(
                    {
                        "signal_date": signal_date,
                        "year": "2022H2" if signal_date[:4] == "2022" else signal_date[:4],
                        "stage": stage_name,
                        "slippage_each_side": slippage,
                        "selected_count": int(len(selected)),
                        "gross_mean": float(np.mean(gross)),
                        "net_mean": float(np.mean(net)),
                    }
                )
            overlap_rows.append(
                {
                    "signal_date": signal_date,
                    "stage": stage_name,
                    "raw_top3_retained": int(np.isin(raw_top, selected).sum()),
                    "selected_count": int(len(selected)),
                }
            )

    detail = pd.DataFrame(rows)
    overlap = pd.DataFrame(overlap_rows)
    summary_rows = []
    for (year, stage, slippage), part in detail.groupby(["year", "stage", "slippage_each_side"], sort=False):
        item = summarize(part["net_mean"].to_numpy(dtype=float))
        summary_rows.append(
            {
                "year": year,
                "stage": stage,
                "slippage_each_side": slippage,
                "days": item["days"],
                "gross_mean": float(part["gross_mean"].mean()),
                "net_mean": item["mean_return"],
                "positive_day_ratio": item["positive_ratio"],
                "avg_selected": float(part["selected_count"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    overlap_summary = (
        overlap.groupby("stage", sort=False)
        .agg(days=("signal_date", "count"), avg_raw_top3_retained=("raw_top3_retained", "mean"), avg_selected=("selected_count", "mean"))
        .reset_index()
    )

    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "daily_top3_filter_ablation.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "year_filter_ablation_summary.csv", index=False, encoding="utf-8-sig")
    overlap_summary.to_csv(OUT / "raw_top3_retention_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "status": "research_only",
        "selection_uses_2026": False,
        "model": "active formal L4 10D rank",
        "signal_dates": [str(dates[0]), str(dates[-1])],
        "execution": "T+1 raw open buy; approximately 10D horizon raw open sell; qfq is not used for execution",
        "costs": {"commission_each_side": 0.0003, "stamp_duty_sell": 0.0005, "slippage_each_side_tested": [0.001, 0.003]},
        "top_n": 3,
        "future_sell_offset_from_buy_open_array": 11,
        "production_change_allowed": False,
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_json(orient="records", force_ascii=False))
    print(overlap_summary.to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    main()
