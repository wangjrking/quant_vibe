from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260713"
POOL_PATH = REPORT_DIR / "four_year_daily_batched_l4_candidate_pool.csv"
OUT_ALL = REPORT_DIR / "rebuild_basefirst_refill_four_year_grid_all.csv"
OUT_TOP = REPORT_DIR / "rebuild_basefirst_refill_four_year_grid_top50.csv"
OUT_SUMMARY = REPORT_DIR / "rebuild_basefirst_refill_four_year_grid_summary.json"
SIGNAL_DIR = REPORT_DIR / "signals"

FULL_START_BUY_DATE = 20220607
FULL_END_BUY_DATE = 20260713
SLIPPAGE_ROUND_TRIP = 0.006
POOL_RET_H1_ALREADY_NET_OF_SLIPPAGE = True


def annualized_return(total_return: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    years = max((end - start).days / 365.25, 1e-9)
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(abs(dd.min()))


def sharpe_ratio(daily_ret: pd.Series) -> float:
    std = float(daily_ret.std(ddof=0))
    if std <= 0:
        return 0.0
    return float(daily_ret.mean() / std * np.sqrt(252.0))


def load_pool() -> pd.DataFrame:
    usecols = [
        "signal_date",
        "buy_date",
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
        "ret_h1",
    ]
    df = pd.read_csv(POOL_PATH, usecols=usecols)
    for col in ["signal_date", "buy_date"]:
        df[col] = df[col].astype(int)
    numeric_cols = [
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
        "ret_h1",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ret_h1", "rank_1d", "rank_3d", "rank_10d"])
    df["w25_25_50"] = 0.25 * df["rank_1d"] + 0.25 * df["rank_3d"] + 0.50 * df["rank_10d"]
    df["w10_70_20"] = 0.10 * df["rank_3d"] + 0.20 * df["rank_5d"] + 0.70 * df["rank_10d"]
    return df


def select_base(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = (
        (df["sig_pct_chg"] <= cfg["sig_pct_max"])
        & (df["buy_open_gap_raw_pct"] <= cfg["gap_high"])
        & (df["buy_open_gap_raw_pct"] >= cfg["gap_low"])
        & (df["rank_10d"] >= cfg["rank10_min"])
        & (df["pred_1d"] >= cfg["pred1_min"])
        & (df["amount"] >= cfg["amount_min"])
        & (df["total_mv"] >= cfg["mv_min"])
    )
    base = df.loc[m].copy()
    if base.empty:
        return base
    base = base.sort_values(["buy_date", cfg["score_col"], "amount"], ascending=[True, False, False])
    base["slot_rank"] = base.groupby("buy_date").cumcount() + 1
    base = base[base["slot_rank"] <= cfg["topn"]].copy()
    base["leg"] = "base"
    base["target_position_pct"] = cfg["base_target"]
    return base


def add_refill(df: pd.DataFrame, base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if cfg["refill"] == "none" or base.empty:
        return base
    per_day = base.groupby("buy_date").size()
    need_days = per_day[per_day < cfg["topn"]].index.tolist()
    if not need_days:
        return base

    if cfg["refill"] == "mv45_deep82":
        refill_mask = (df["sig_pct_chg"] <= -8.2) & (df["total_mv"] <= 450000)
    elif cfg["refill"] == "mv83_deep82":
        refill_mask = (df["sig_pct_chg"] <= -8.2) & (df["total_mv"] <= 830000)
    elif cfg["refill"] == "turn45_deep82":
        refill_mask = (df["sig_pct_chg"] <= -8.2) & (df["turnover_rate"] >= 4.5)
    else:
        refill_mask = pd.Series(False, index=df.index)

    chosen_keys = set(zip(base["buy_date"], base["stock_code"]))
    refill = df.loc[refill_mask & df["buy_date"].isin(need_days)].copy()
    if refill.empty:
        return base
    refill = refill[~list(zip(refill["buy_date"], refill["stock_code"])).__contains__] if False else refill
    refill = refill[~refill.apply(lambda r: (r["buy_date"], r["stock_code"]) in chosen_keys, axis=1)]
    refill = refill.sort_values(["buy_date", cfg["score_col"], "amount"], ascending=[True, False, False])
    rows = []
    for buy_date, part in refill.groupby("buy_date", sort=True):
        need = int(cfg["topn"] - per_day.get(buy_date, 0))
        if need <= 0:
            continue
        rows.append(part.head(need))
    if not rows:
        return base
    refill_out = pd.concat(rows, ignore_index=True)
    refill_out["leg"] = "refill"
    refill_out["target_position_pct"] = cfg["refill_target"]
    return pd.concat([base, refill_out], ignore_index=True)


def apply_weight_scheme(sig: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if sig.empty:
        return sig
    out = sig.copy()
    out["weight_scale"] = 1.0
    if cfg["scale"] == "weak85_strong105":
        weak = (out["sig_pct_chg"] >= -2.5) | (out["buy_open_gap_raw_pct"] >= 0.0)
        strong = (
            (out["sig_pct_chg"] <= -5.0)
            & (out["buy_open_gap_raw_pct"] <= -0.8)
            & (out["turnover_rate"] >= 4.5)
        )
        out.loc[weak, "weight_scale"] = 0.85
        out.loc[strong, "weight_scale"] = 1.05
    elif cfg["scale"] == "weak90_strong110":
        weak = (out["sig_pct_chg"] >= -2.5) | (out["buy_open_gap_raw_pct"] >= 0.0)
        strong = (
            (out["sig_pct_chg"] <= -5.0)
            & (out["buy_open_gap_raw_pct"] <= -0.8)
            & (out["turnover_rate"] >= 4.5)
        )
        out.loc[weak, "weight_scale"] = 0.90
        out.loc[strong, "weight_scale"] = 1.10
    out["target_position_pct"] = out["target_position_pct"] * out["weight_scale"]

    # Cap each day's total target exposure. This avoids unstable oversubscription.
    totals = out.groupby("buy_date")["target_position_pct"].transform("sum")
    cap = cfg["daily_cap"]
    scale = np.where(totals > cap, cap / totals, 1.0)
    out["target_position_pct"] = out["target_position_pct"] * scale
    return out


def evaluate(sig: pd.DataFrame, cfg: dict, trade_dates: list[int]) -> dict:
    full_index = pd.to_datetime([str(d) for d in trade_dates], format="%Y%m%d")
    if sig.empty:
        daily = pd.Series(0.0, index=full_index)
        equity = (1.0 + daily).cumprod()
        return {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "trades": 0,
            "buy_days": 0,
            "win_ratio": 0.0,
            "avg_daily_target_sum": 0.0,
            "max_daily_target_sum": 0.0,
            "min_year_annual": 0.0,
            "median_year_annual": 0.0,
        }
    work = sig.copy()
    if POOL_RET_H1_ALREADY_NET_OF_SLIPPAGE:
        work["net_ret"] = work["ret_h1"]
    else:
        work["net_ret"] = work["ret_h1"] - SLIPPAGE_ROUND_TRIP
    work["weighted_ret"] = work["target_position_pct"] * work["net_ret"]
    by_day = work.groupby("buy_date").agg(
        daily_ret=("weighted_ret", "sum"),
        target_sum=("target_position_pct", "sum"),
    )
    daily = pd.Series(0.0, index=pd.Index(trade_dates, name="buy_date"), dtype=float)
    daily.loc[by_day.index.intersection(daily.index)] = by_day["daily_ret"]
    daily_dt = pd.Series(daily.to_numpy(), index=full_index)
    equity = (1.0 + daily_dt).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    year_returns = []
    for year, part in daily_dt.groupby(daily_dt.index.year):
        if part.empty:
            continue
        eq = (1.0 + part).cumprod()
        year_returns.append(float(eq.iloc[-1] - 1.0))
    return {
        "annual_return": annualized_return(total_return, full_index[0], full_index[-1]),
        "sharpe": sharpe_ratio(daily_dt),
        "max_drawdown": max_drawdown(equity),
        "total_return": total_return,
        "trades": int(len(work)),
        "buy_days": int(work["buy_date"].nunique()),
        "win_ratio": float((work["net_ret"] > 0).mean()),
        "avg_daily_target_sum": float(by_day["target_sum"].mean()),
        "max_daily_target_sum": float(by_day["target_sum"].max()),
        "min_year_annual": float(min(year_returns)) if year_returns else 0.0,
        "median_year_annual": float(np.median(year_returns)) if year_returns else 0.0,
        "recent60_return": float((1.0 + daily_dt.iloc[-60:]).prod() - 1.0),
        "recent120_return": float((1.0 + daily_dt.iloc[-120:]).prod() - 1.0),
    }


def make_variant_name(cfg: dict) -> str:
    def f(x):
        return str(x).replace("-", "m").replace(".", "p")

    return (
        f"{cfg['score_col']}_top{cfg['topn']}_bt{f(cfg['base_target'])}_rt{f(cfg['refill_target'])}"
        f"_p1{f(cfg['pred1_min'])}_r10{f(cfg['rank10_min'])}_pct{f(cfg['sig_pct_max'])}"
        f"_gap{f(cfg['gap_low'])}_{f(cfg['gap_high'])}_amt{int(cfg['amount_min'])}"
        f"_mv{int(cfg['mv_min'])}_{cfg['refill']}_{cfg['scale']}_cap{f(cfg['daily_cap'])}"
    )


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pool()
    trade_dates = sorted(int(x) for x in df["buy_date"].dropna().unique() if FULL_START_BUY_DATE <= int(x) <= FULL_END_BUY_DATE)

    cfgs = []
    for (
        score_col,
        topn,
        base_target,
        refill_target,
        pred1_min,
        rank10_min,
        sig_pct_max,
        gap_low,
        gap_high,
        amount_min,
        mv_min,
        refill,
        scale,
        daily_cap,
    ) in itertools.product(
        ["w25_25_50", "w10_70_20"],
        [2, 3],
        [0.40, 0.435],
        [0.10],
        [-0.002],
        [0.70],
        [-1.75, -2.0],
        [-10.0, -8.0],
        [1.5],
        [90000],
        [200000, 800000],
        ["none", "mv45_deep82", "mv83_deep82"],
        ["none", "weak85_strong105"],
        [0.91],
    ):
        cfgs.append(
            {
                "score_col": score_col,
                "topn": topn,
                "base_target": base_target,
                "refill_target": refill_target,
                "pred1_min": pred1_min,
                "rank10_min": rank10_min,
                "sig_pct_max": sig_pct_max,
                "gap_low": gap_low,
                "gap_high": gap_high,
                "amount_min": amount_min,
                "mv_min": mv_min,
                "refill": refill,
                "scale": scale,
                "daily_cap": daily_cap,
            }
        )

    rows = []
    best_signals: list[tuple[str, pd.DataFrame, dict]] = []
    for idx, cfg in enumerate(cfgs, start=1):
        base = select_base(df, cfg)
        sig = add_refill(df, base, cfg)
        sig = apply_weight_scheme(sig, cfg)
        stats = evaluate(sig, cfg, trade_dates)
        variant = make_variant_name(cfg)
        score = (
            stats["annual_return"]
            + 0.25 * stats["sharpe"]
            - 2.0 * stats["max_drawdown"]
            + 0.25 * min(stats["min_year_annual"], 0.0)
        )
        row = {"variant": variant, **cfg, **stats, "rank_score": score}
        rows.append(row)
        if len(best_signals) < 20 or score > min(x[2]["rank_score"] for x in best_signals):
            best_signals.append((variant, sig, row))
            best_signals = sorted(best_signals, key=lambda x: x[2]["rank_score"], reverse=True)[:20]
        if idx % 5000 == 0:
            print(f"processed {idx}/{len(cfgs)}")

    result = pd.DataFrame(rows).sort_values(
        ["annual_return", "sharpe", "max_drawdown"], ascending=[False, False, True]
    )
    result.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    result.head(50).to_csv(OUT_TOP, index=False, encoding="utf-8-sig")

    top_by_rank = sorted(best_signals, key=lambda x: x[2]["rank_score"], reverse=True)[:5]
    saved_signals = []
    for variant, sig, stats in top_by_rank:
        out_path = SIGNAL_DIR / f"{variant[:180]}.csv"
        cols = [
            "signal_date",
            "buy_date",
            "stock_code",
            "name",
            "leg",
            "target_position_pct",
            "sig_pct_chg",
            "buy_open_gap_raw_pct",
            "turnover_rate",
            "amount",
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
        ]
        sig.sort_values(["buy_date", "target_position_pct", "stock_code"], ascending=[True, False, True])[cols].to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
        saved_signals.append({"variant": variant, "path": str(out_path), **stats})

    summary = {
        "status": "local_research_proxy_only",
        "created_at": "2026-07-13",
        "pool_path": str(POOL_PATH),
        "pool_rows": int(len(df)),
        "pool_signal_date_min": int(df["signal_date"].min()),
        "pool_signal_date_max": int(df["signal_date"].max()),
        "pool_buy_date_min": int(df["buy_date"].min()),
        "pool_buy_date_max": int(df["buy_date"].max()),
        "required_full_start_buy_date": FULL_START_BUY_DATE,
        "required_full_end_buy_date": FULL_END_BUY_DATE,
        "trade_days_used": len(trade_dates),
        "grid_rows": int(len(result)),
        "target_hit_count": int(((result["annual_return"] >= 5.0) & (result["sharpe"] >= 4.0) & (result["max_drawdown"] <= 0.40)).sum()),
        "top20_by_annual": result.head(20).to_dict(orient="records"),
        "top5_saved_signals": saved_signals,
        "formal_validation_requirement": "Juejin backtest required before any production or admission claim",
        "slippage_round_trip": SLIPPAGE_ROUND_TRIP,
        "pool_ret_h1_already_net_of_slippage": POOL_RET_H1_ALREADY_NET_OF_SLIPPAGE,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "grid_rows", "target_hit_count", "pool_signal_date_min", "pool_signal_date_max"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
