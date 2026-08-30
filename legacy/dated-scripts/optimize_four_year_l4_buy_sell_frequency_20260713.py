from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260713"
POOL_PATH = REPORT_DIR / "four_year_daily_batched_l4_candidate_pool.csv"
OUT_ALL = REPORT_DIR / "four_year_buy_sell_frequency_grid_corrected_all.csv"
OUT_TOP = REPORT_DIR / "four_year_buy_sell_frequency_grid_corrected_top100.csv"
OUT_SUMMARY = REPORT_DIR / "four_year_buy_sell_frequency_grid_corrected_summary.json"
SIGNAL_DIR = REPORT_DIR / "signals" / "buy_sell_frequency_corrected"

FULL_START_BUY_DATE = 20220607
FULL_END_BUY_DATE = 20260713


def annualized_return(total_return: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    years = max((end - start).days / 365.25, 1e-9)
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(abs((equity / peak - 1.0).min()))


def sharpe_ratio(daily_ret: pd.Series) -> float:
    std = float(daily_ret.std(ddof=0))
    if std <= 0:
        return 0.0
    return float(daily_ret.mean() / std * np.sqrt(252.0))


def load_pool() -> pd.DataFrame:
    usecols = [
        "signal_date",
        "buy_date",
        "sell_h1_date",
        "sell_h2_date",
        "sell_h3_date",
        "stock_code",
        "name",
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "rank_1d",
        "rank_3d",
        "rank_5d",
        "rank_10d",
        "sig_pct_chg",
        "buy_open_gap_raw_pct",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "ret_h1",
        "ret_h2",
        "ret_h3",
    ]
    df = pd.read_csv(POOL_PATH, usecols=usecols)
    for c in ["signal_date", "buy_date", "sell_h1_date", "sell_h2_date", "sell_h3_date"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    num_cols = [c for c in usecols if c not in {"stock_code", "name"}]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["signal_date", "buy_date", "ret_h1", "rank_10d"])
    df["signal_date"] = df["signal_date"].astype(int)
    df["buy_date"] = df["buy_date"].astype(int)
    df["score_10d100"] = df["rank_10d"]
    df["score_10d85_5d10_3d5"] = 0.85 * df["rank_10d"] + 0.10 * df["rank_5d"] + 0.05 * df["rank_3d"]
    df["score_10d70_5d20_3d10"] = 0.70 * df["rank_10d"] + 0.20 * df["rank_5d"] + 0.10 * df["rank_3d"]
    df["score_1d25_3d25_10d50"] = 0.25 * df["rank_1d"] + 0.25 * df["rank_3d"] + 0.50 * df["rank_10d"]
    df["score_1d20_10d80"] = 0.20 * df["rank_1d"] + 0.80 * df["rank_10d"]
    return df


def evaluate(sig: pd.DataFrame, ret_col: str, target_mode: str, exposure: float, trade_dates: list[int]) -> dict:
    full_index = pd.to_datetime([str(d) for d in trade_dates], format="%Y%m%d")
    if sig.empty:
        daily_ret = pd.Series(0.0, index=full_index)
        equity = (1.0 + daily_ret).cumprod()
        return {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "trades": 0,
            "buy_days": 0,
            "win_ratio": 0.0,
            "avg_daily_names": 0.0,
            "avg_daily_target_sum": 0.0,
            "max_daily_target_sum": 0.0,
            "min_year_annual": 0.0,
            "median_year_annual": 0.0,
            "recent60_return": 0.0,
            "recent120_return": 0.0,
        }
    work = sig.copy()
    if target_mode == "equal":
        counts = work.groupby("buy_date")["stock_code"].transform("count")
        work["target_position_pct"] = exposure / counts
    elif target_mode == "rank_taper":
        rank = work.groupby("buy_date").cumcount() + 1
        raw = np.where(rank == 1, 0.45, np.where(rank == 2, 0.30, np.where(rank == 3, 0.20, 0.10)))
        work["raw_target"] = raw
        total = work.groupby("buy_date")["raw_target"].transform("sum")
        work["target_position_pct"] = exposure * work["raw_target"] / total
    else:
        raise ValueError(target_mode)
    work["weighted_ret"] = work["target_position_pct"] * work[ret_col]
    by_day = work.groupby("buy_date").agg(
        daily_ret=("weighted_ret", "sum"),
        names=("stock_code", "count"),
        target_sum=("target_position_pct", "sum"),
    )
    daily = pd.Series(0.0, index=pd.Index(trade_dates, name="buy_date"), dtype=float)
    daily.loc[by_day.index.intersection(daily.index)] = by_day["daily_ret"]
    daily_dt = pd.Series(daily.to_numpy(), index=full_index)
    equity = (1.0 + daily_dt).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    year_returns = []
    for _, part in daily_dt.groupby(daily_dt.index.year):
        if len(part):
            year_returns.append(float((1.0 + part).prod() - 1.0))
    return {
        "annual_return": annualized_return(total_return, full_index[0], full_index[-1]),
        "sharpe": sharpe_ratio(daily_dt),
        "max_drawdown": max_drawdown(equity),
        "total_return": total_return,
        "trades": int(len(work)),
        "buy_days": int(work["buy_date"].nunique()),
        "win_ratio": float((work[ret_col] > 0).mean()),
        "avg_daily_names": float(by_day["names"].mean()),
        "avg_daily_target_sum": float(by_day["target_sum"].mean()),
        "max_daily_target_sum": float(by_day["target_sum"].max()),
        "min_year_annual": float(min(year_returns)) if year_returns else 0.0,
        "median_year_annual": float(np.median(year_returns)) if year_returns else 0.0,
        "recent60_return": float((1.0 + daily_dt.iloc[-60:]).prod() - 1.0),
        "recent120_return": float((1.0 + daily_dt.iloc[-120:]).prod() - 1.0),
    }


def build_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    score_col = cfg["score_col"]
    m = (
        (df["amount"] >= cfg["amount_min"])
        & (df["total_mv"] >= cfg["mv_min"])
        & (df["buy_open_gap_raw_pct"] >= cfg["gap_min"])
        & (df["buy_open_gap_raw_pct"] <= cfg["gap_max"])
        & (df["sig_pct_chg"] <= cfg["sig_pct_max"])
        & (df["turnover_rate"] >= cfg["turnover_min"])
        & (df["rank_10d"] >= cfg["rank10_min"])
        & (df[score_col] >= cfg["score_min"])
    )
    if cfg["atr_max"] is not None:
        m &= df["atr_qfq"].notna() & (df["atr_qfq"] <= cfg["atr_max"])
    part = df.loc[m].copy()
    if part.empty:
        return part
    part = part.dropna(subset=[cfg["ret_col"]])
    part = part.sort_values(["buy_date", score_col, "amount"], ascending=[True, False, False])
    part["daily_rank"] = part.groupby("buy_date").cumcount() + 1
    part = part[part["daily_rank"] <= cfg["topn"]].copy()
    return part


def variant_name(cfg: dict) -> str:
    def f(v):
        return str(v).replace("-", "m").replace(".", "p").replace("None", "na")

    return (
        f"{cfg['score_col']}_top{cfg['topn']}_{cfg['ret_col']}_{cfg['target_mode']}"
        f"_ex{f(cfg['exposure'])}_smin{f(cfg['score_min'])}_r10{f(cfg['rank10_min'])}"
        f"_pct{f(cfg['sig_pct_max'])}_gap{f(cfg['gap_min'])}_{f(cfg['gap_max'])}"
        f"_amt{int(cfg['amount_min'])}_mv{int(cfg['mv_min'])}_turn{f(cfg['turnover_min'])}_atr{f(cfg['atr_max'])}"
    )


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pool()
    trade_dates = sorted(int(x) for x in df["buy_date"].dropna().unique() if FULL_START_BUY_DATE <= int(x) <= FULL_END_BUY_DATE)

    score_cols = [
        "score_10d100",
        "score_10d85_5d10_3d5",
    ]
    cfgs = []
    for values in itertools.product(
        score_cols,
        ["ret_h1", "ret_h2", "ret_h3"],
        ["equal"],
        [3, 5],
        [0.9, 1.0],
        [0.90],
        [0.70, 0.90],
        [-1.75, -3.0, -5.0],
        [-5.0, -3.0],
        [0.5],
        [300000, 500000],
        [800000],
        [0.0, 4.5],
        [None],
    ):
        (
            score_col,
            ret_col,
            target_mode,
            topn,
            exposure,
            score_min,
            rank10_min,
            sig_pct_max,
            gap_min,
            gap_max,
            amount_min,
            mv_min,
            turnover_min,
            atr_max,
        ) = values
        cfgs.append(
            {
                "score_col": score_col,
                "ret_col": ret_col,
                "target_mode": target_mode,
                "topn": topn,
                "exposure": exposure,
                "score_min": score_min,
                "rank10_min": rank10_min,
                "sig_pct_max": sig_pct_max,
                "gap_min": gap_min,
                "gap_max": gap_max,
                "amount_min": amount_min,
                "mv_min": mv_min,
                "turnover_min": turnover_min,
                "atr_max": atr_max,
            }
        )

    rows = []
    top_saved: list[tuple[float, str, pd.DataFrame, dict]] = []
    for i, cfg in enumerate(cfgs, 1):
        sig = build_signals(df, cfg)
        stats = evaluate(sig, cfg["ret_col"], cfg["target_mode"], cfg["exposure"], trade_dates)
        # Penalize tiny samples and unstable year splits.
        stability_penalty = 0.0
        if stats["buy_days"] < 120:
            stability_penalty += (120 - stats["buy_days"]) / 120
        if stats["min_year_annual"] < -0.20:
            stability_penalty += abs(stats["min_year_annual"] + 0.20)
        rank_score = stats["annual_return"] + 0.35 * stats["sharpe"] - 1.5 * stats["max_drawdown"] - stability_penalty
        variant = variant_name(cfg)
        row = {"variant": variant, **cfg, **stats, "rank_score": rank_score}
        rows.append(row)
        if len(sig) and (len(top_saved) < 10 or rank_score > min(x[0] for x in top_saved)):
            top_saved.append((rank_score, variant, sig, row))
            top_saved = sorted(top_saved, key=lambda x: x[0], reverse=True)[:10]
        if i % 20000 == 0:
            print(f"processed {i}/{len(cfgs)}")

    result = pd.DataFrame(rows)
    result = result.sort_values(["annual_return", "sharpe", "max_drawdown"], ascending=[False, False, True])
    result.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    result.head(100).to_csv(OUT_TOP, index=False, encoding="utf-8-sig")

    saved = []
    for _, variant, sig, row in top_saved:
        path = SIGNAL_DIR / f"{variant[:180]}.csv"
        out = sig.copy()
        out["variant"] = variant
        cols = [
            "variant",
            "signal_date",
            "buy_date",
            "stock_code",
            "name",
            "daily_rank",
            "sig_pct_chg",
            "buy_open_gap_raw_pct",
            "amount",
            "turnover_rate",
            "total_mv",
            "rank_1d",
            "rank_3d",
            "rank_5d",
            "rank_10d",
            "pred_1d",
            "pred_3d",
            "pred_5d",
            "pred_10d",
            "ret_h1",
            "ret_h2",
            "ret_h3",
        ]
        out.sort_values(["buy_date", "daily_rank"]).to_csv(path, index=False, columns=cols, encoding="utf-8-sig")
        saved.append({"variant": variant, "path": str(path), **row})

    summary = {
        "status": "local_research_proxy_only",
        "created_at": "2026-07-13",
        "pool_path": str(POOL_PATH),
        "pool_rows": int(len(df)),
        "pool_signal_date_min": int(df["signal_date"].min()),
        "pool_signal_date_max": int(df["signal_date"].max()),
        "pool_buy_date_min": int(df["buy_date"].min()),
        "pool_buy_date_max": int(df["buy_date"].max()),
        "full_window_buy_start": FULL_START_BUY_DATE,
        "full_window_buy_end": FULL_END_BUY_DATE,
        "trade_days_used": len(trade_dates),
        "grid_rows": int(len(result)),
        "target_hit_count": int(((result["annual_return"] >= 5.0) & (result["sharpe"] >= 4.0) & (result["max_drawdown"] <= 0.40)).sum()),
        "admission_like_count": int(((result["annual_return"] > 0) & (result["sharpe"] > 1.0) & (result["max_drawdown"] <= 0.40) & (result["buy_days"] >= 120)).sum()),
        "top20_by_annual": result.head(20).to_dict(orient="records"),
        "top10_saved_by_rank_score": saved,
        "note": "ret_h1/ret_h2/ret_h3 are used as stored in the candidate pool; do not double subtract slippage.",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "grid_rows", "target_hit_count", "admission_like_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
