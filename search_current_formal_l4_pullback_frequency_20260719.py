from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_smooth_frequency_20260719"
POOL_DB = REPORT_DIR / "current_formal_l4_rank_pool.duckdb"
OUT_CSV = REPORT_DIR / "pullback_frequency_local_grid.csv"
OUT_JSON = REPORT_DIR / "pullback_frequency_local_grid.json"
TOP_DIR = REPORT_DIR / "pullback_top_signals"

BLENDS = {
    "b10_100": {"rank_1d": 0.0, "rank_3d": 0.0, "rank_5d": 0.0, "rank_10d": 1.0},
    "b5_20_10_80": {"rank_1d": 0.0, "rank_3d": 0.0, "rank_5d": 0.20, "rank_10d": 0.80},
    "b3_10_5_20_10_70": {"rank_1d": 0.0, "rank_3d": 0.10, "rank_5d": 0.20, "rank_10d": 0.70},
    "b1_05_3_10_5_15_10_70": {"rank_1d": 0.05, "rank_3d": 0.10, "rank_5d": 0.15, "rank_10d": 0.70},
}


def period_stats(daily: pd.Series) -> tuple[float, float, float]:
    if daily.empty:
        return 0.0, 0.0, 0.0
    equity = (1.0 + daily).cumprod()
    annual = float(equity.iloc[-1] ** (252.0 / len(daily)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=0))
    sharpe = float(daily.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    mdd = float(abs((equity / equity.cummax() - 1.0).min()))
    return annual, sharpe, mdd


def score_frame(frame: pd.DataFrame, weights: dict[str, float], drop_bonus: float, liq_bonus: float) -> np.ndarray:
    score = np.zeros(len(frame), dtype=float)
    for column, weight in weights.items():
        score += float(weight) * frame[column].to_numpy(float)
    score += float(drop_bonus) * np.clip((-frame["signal_pct_chg_raw"].to_numpy(float) - 1.0) / 8.0, 0.0, 1.0)
    score += float(liq_bonus) * (frame["amount_rank"].to_numpy(float) + frame["mv_rank"].to_numpy(float)) / 2.0
    return score


def evaluate(selected: pd.DataFrame, topn: int, sizing: str, all_dates: pd.Index) -> dict:
    work = selected.copy()
    if sizing == "smooth":
        positive_gap = np.maximum(work["buy_open_gap_raw_pct"].to_numpy(float) - 0.5, 0.0)
        deep_drop = np.maximum(-4.0 - work["signal_pct_chg_raw"].to_numpy(float), 0.0)
        weak_signal = np.maximum(work["signal_pct_chg_raw"].to_numpy(float) + 2.0, 0.0)
        scale = np.clip(1.0 - 0.05 * positive_gap + 0.15 * deep_drop - 0.35 * weak_signal, 0.20, 1.80)
    else:
        scale = np.ones(len(work), dtype=float)
    work["target_pct"] = 0.90 / topn * scale
    daily_target = work.groupby("buy_date")["target_pct"].transform("sum")
    work["target_pct"] *= np.minimum(1.0, 0.90 / daily_target.replace(0.0, np.nan)).fillna(1.0)
    work["weighted_return"] = work["target_pct"] * (work["next_open_return_raw"] - 0.006)
    daily = work.groupby("buy_date")["weighted_return"].sum().reindex(all_dates, fill_value=0.0)
    annual, sharpe, mdd = period_stats(daily)
    dated = daily.copy()
    dated.index = pd.to_datetime(dated.index)
    yearly = [period_stats(group) for _, group in dated.groupby(dated.index.year) if len(group) >= 40]
    rolling = [period_stats(daily.iloc[start : start + 60]) for start in range(0, max(len(daily) - 59, 1), 20)]
    return {
        "local_annual": annual,
        "local_sharpe": sharpe,
        "local_mdd": mdd,
        "worst_year_annual": min((item[0] for item in yearly), default=0.0),
        "worst_year_sharpe": min((item[1] for item in yearly), default=0.0),
        "rolling60_min_annual": min((item[0] for item in rolling), default=0.0),
        "rolling60_min_sharpe": min((item[1] for item in rolling), default=0.0),
        "rolling60_median_sharpe": float(np.median([item[1] for item in rolling])) if rolling else 0.0,
        "signal_days": int(work["trade_date"].nunique()),
        "rows": int(len(work)),
        "mean_daily_target": float(work.groupby("buy_date")["target_pct"].sum().mean()),
    }


def write_signal(frame: pd.DataFrame, row: dict) -> Path:
    work = frame.copy()
    topn = int(row["topn"])
    if row["sizing"] == "smooth":
        positive_gap = np.maximum(work["buy_open_gap_raw_pct"].to_numpy(float) - 0.5, 0.0)
        deep_drop = np.maximum(-4.0 - work["signal_pct_chg_raw"].to_numpy(float), 0.0)
        weak_signal = np.maximum(work["signal_pct_chg_raw"].to_numpy(float) + 2.0, 0.0)
        scale = np.clip(1.0 - 0.05 * positive_gap + 0.15 * deep_drop - 0.35 * weak_signal, 0.20, 1.80)
    else:
        scale = np.ones(len(work), dtype=float)
    work["target_pct"] = 0.90 / topn * scale
    daily_target = work.groupby("buy_date")["target_pct"].transform("sum")
    work["target_pct"] *= np.minimum(1.0, 0.90 / daily_target.replace(0.0, np.nan)).fillna(1.0)
    work["signal_date"] = work["trade_date"].astype(str)
    work["symbol"] = np.where(
        work["stock_code"].str.endswith(".SH"),
        "SHSE." + work["stock_code"].str[:6],
        "SZSE." + work["stock_code"].str[:6],
    )
    work["rank"] = work["pick_rank"].astype(int)
    work["pred_prob"] = work["select_score"]
    work["entry_score"] = work["select_score"]
    work["holding_days"] = 1
    work["max_holding_days"] = 3
    work["score_exit_entry_ratio"] = 0.98
    work["min_holding_days_before_score_exit"] = 1
    work["score_continue_entry_ratio"] = 1.02
    work["strategy_variant"] = row["name"]
    work["filter_name"] = "current_formal_l4_pullback_frequency"
    work["entry_weight_name"] = row["sizing"]
    work["dynamic_hold_name"] = "h1_mh3_score_exit"
    work["buy_day_market_available"] = True
    work["buy_day_hard_gate_complete"] = True
    work["buy_day_st_rejected"] = False
    work["buy_day_open_limit_up_rejected"] = False
    work["latest_market_date"] = str(work["buy_date"].max())
    work["buy_open_gap_pct"] = work["buy_open_gap_raw_pct"]
    columns = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "rank_1d", "rank_3d", "rank_5d", "rank_10d",
        "amount", "turnover_rate", "total_mv", "atr_qfq", "signal_pct_chg_raw", "target_pct",
        "holding_days", "max_holding_days", "score_exit_entry_ratio", "min_holding_days_before_score_exit",
        "score_continue_entry_ratio", "strategy_variant", "filter_name", "entry_weight_name", "dynamic_hold_name",
        "buy_day_market_available", "buy_day_hard_gate_complete", "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected", "latest_market_date", "buy_open_gap_pct", "buy_open_gap_raw_pct",
    ]
    path = TOP_DIR / f"{row['name']}.csv"
    work[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    TOP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(POOL_DB), read_only=True)
    frame = con.execute(
        """
        select *
        from current_formal_l4_rank_pool
        where amount >= 90000
          and total_mv >= 200000
          and signal_pct_chg_raw <= 0
          and next_open_return_raw is not null
        """
    ).fetchdf()
    con.close()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["buy_date"] = frame["buy_date"].astype(str)
    all_dates = pd.date_range(pd.to_datetime(frame["buy_date"].min()), pd.to_datetime(frame["buy_date"].max()), freq="B").strftime("%Y%m%d")
    rows = []
    for blend, weights in BLENDS.items():
        for drop_bonus in [0.0, 0.02]:
            for liq_bonus in [0.01]:
                scored = frame.copy()
                scored["model_score"] = score_frame(scored, weights, 0.0, 0.0)
                drop_component = drop_bonus * np.clip((-scored["signal_pct_chg_raw"].to_numpy(float) - 1.0) / 8.0, 0.0, 1.0)
                liq_component = liq_bonus * (scored["amount_rank"].to_numpy(float) + scored["mv_rank"].to_numpy(float)) / 2.0
                for score_cap in [0.90, 0.95, 0.98, 1.00]:
                    scored["select_score"] = np.minimum(scored["model_score"], score_cap) + drop_component + liq_component
                    for model_min in [0.70, 0.80, 0.90]:
                        for pct_max in [0.0, -1.0, -2.0, -3.0, -4.0]:
                            eligible = scored[
                                (scored["signal_pct_chg_raw"] <= pct_max)
                                & (scored["model_score"] >= model_min)
                            ].copy()
                            eligible = eligible.sort_values(["trade_date", "select_score", "stock_code"], ascending=[True, False, True])
                            eligible["pick_rank"] = eligible.groupby("trade_date").cumcount() + 1
                            for topn in [2, 3, 5]:
                                selected = eligible[eligible["pick_rank"] <= topn].copy()
                                for sizing in ["plain", "smooth"]:
                                    name = (
                                        f"{blend}_p{abs(int(pct_max))}_m{int(model_min*100)}_c{int(score_cap*100)}_"
                                        f"db{int(drop_bonus*100):02d}_top{topn}_{sizing}"
                                    )
                                    stats = evaluate(selected, topn, sizing, all_dates)
                                    result = {
                                        "name": name,
                                        "blend": blend,
                                        "pct_max": pct_max,
                                        "model_min": model_min,
                                        "score_cap": score_cap,
                                        "drop_bonus": drop_bonus,
                                        "liq_bonus": liq_bonus,
                                        "topn": topn,
                                        "sizing": sizing,
                                        **stats,
                                    }
                                    rows.append(result)
    results = pd.DataFrame(rows)
    results["stability_score"] = (
        results["local_sharpe"]
        + 0.35 * results["worst_year_sharpe"]
        + 0.20 * results["rolling60_median_sharpe"]
        + 0.05 * results["rolling60_min_sharpe"]
    )
    results = results.sort_values(["stability_score", "local_sharpe", "local_annual"], ascending=False)
    results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    top_names = list(dict.fromkeys(results.head(12)["name"].tolist()))
    manifest = []
    for name in top_names:
        row = results[results["name"] == name].iloc[0].to_dict()
        selected = frame.copy()
        selected["select_score"] = score_frame(
            selected,
            BLENDS[str(row["blend"])],
            0.0,
            0.0,
        )
        selected["model_score"] = selected["select_score"]
        selected["select_score"] = (
            np.minimum(selected["model_score"], float(row["score_cap"]))
            + float(row["drop_bonus"]) * np.clip((-selected["signal_pct_chg_raw"] - 1.0) / 8.0, 0.0, 1.0)
            + float(row["liq_bonus"]) * (selected["amount_rank"] + selected["mv_rank"]) / 2.0
        )
        selected = selected[
            (selected["signal_pct_chg_raw"] <= float(row["pct_max"]))
            & (selected["model_score"] >= float(row["model_min"]))
        ].copy()
        selected = selected.sort_values(["trade_date", "select_score", "stock_code"], ascending=[True, False, True])
        selected["pick_rank"] = selected.groupby("trade_date").cumcount() + 1
        selected = selected[selected["pick_rank"] <= int(row["topn"])].copy()
        path = write_signal(selected, row)
        manifest.append({**row, "signal_file": str(path), "score_db": str(REPORT_DIR / "score_assets" / f"{row['blend']}.duckdb"), "score_table": "blended_rank_score"})
    payload = {
        "status": "research_only_local_screen",
        "selection_contract": "current active formal L4 daily ranks; signal-day data only; buy open used for continuous sizing; next open used for evaluation only",
        "case_count": int(len(results)),
        "top_manifest": manifest,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(results.head(20).to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    main()
