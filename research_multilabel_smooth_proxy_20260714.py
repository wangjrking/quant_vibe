from __future__ import annotations

import itertools
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
CANDIDATE_PARQUET = REPORT_DIR / "active_l4_multilabel_pool_candidates_20260714.parquet"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT_CSV = REPORT_DIR / "multilabel_smooth_proxy_search_20260714.csv"
OUT_JSON = REPORT_DIR / "multilabel_smooth_proxy_search_20260714.json"


def load_candidates() -> pd.DataFrame:
    df = pd.read_parquet(CANDIDATE_PARQUET)
    for col in [
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "signal_pct_chg_raw",
        "buy_open_gap_raw_pct",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_returns() -> pd.DataFrame:
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        ret = con.execute(
            """
            with cal as (
              select trade_date,
                     lead(trade_date, 1) over(order by trade_date) as d1,
                     lead(trade_date, 2) over(order by trade_date) as d2,
                     lead(trade_date, 3) over(order by trade_date) as d3
              from (select distinct trade_date from STOCK_DAILY_DATA order by trade_date)
            )
            select
              b.stock_code,
              b.trade_date as buy_date,
              b.open as buy_open,
              e1.open as open_d1,
              e2.open as open_d2,
              e3.open as open_d3
            from STOCK_DAILY_DATA b
            join cal c on c.trade_date=b.trade_date
            left join STOCK_DAILY_DATA e1 on e1.trade_date=c.d1 and e1.stock_code=b.stock_code
            left join STOCK_DAILY_DATA e2 on e2.trade_date=c.d2 and e2.stock_code=b.stock_code
            left join STOCK_DAILY_DATA e3 on e3.trade_date=c.d3 and e3.stock_code=b.stock_code
            where b.trade_date between '20220607' and '20260714'
            """
        ).fetchdf()
    finally:
        con.close()
    ret["buy_date"] = ret["buy_date"].astype(str)
    for h in [1, 2, 3]:
        ret[f"ret_h{h}"] = ret[f"open_d{h}"] / ret["buy_open"] - 1.0
    return ret[["stock_code", "buy_date", "ret_h1", "ret_h2", "ret_h3"]]


def score_case(df: pd.DataFrame, case: dict) -> dict | None:
    w = case["weights"]
    work = df.copy()
    work["entry_score"] = (
        w[0] * work["pred_10d"].fillna(0.0)
        + w[1] * work["pred_5d"].fillna(0.0)
        + w[2] * work["pred_3d"].fillna(0.0)
        + w[3] * work["pred_1d"].fillna(0.0)
    )
    mask = (
        (work["pred_10d"] >= case["p10_min"])
        & (work["pred_5d"] >= case["p5_min"])
        & (work["pred_1d"] >= case["p1_min"])
        & (work["signal_pct_chg_raw"] <= case["pct_max"])
        & (work["signal_pct_chg_raw"] >= case["pct_min"])
        & (work["amount"] >= case["amount_min"])
        & (work["total_mv"] >= case["mv_min"])
        & (work["atr_qfq"] <= case["atr_max"])
        & (work["buy_open_gap_raw_pct"] <= case["gap_max"])
        & (work["buy_open_gap_raw_pct"] >= case["gap_min"])
    )
    work = work.loc[mask].copy()
    if len(work) < 200:
        return None
    work["liquidity_score"] = work["amount"].rank(pct=True) + work["total_mv"].rank(pct=True)
    work["sort_score"] = work["entry_score"] + case["liq_bonus"] * work["liquidity_score"]
    work = (
        work.sort_values(["buy_date", "sort_score", "pred_10d"], ascending=[True, False, False])
        .groupby("buy_date", group_keys=False)
        .head(case["top_n"])
        .copy()
    )
    if work["buy_date"].nunique() < 180:
        return None
    base_target = min(case["cap"] / case["top_n"], case["base_target"])
    work["target_pct"] = base_target
    high = (work["entry_score"] >= case["high_score"]) & (work["signal_pct_chg_raw"] <= case["deep_pct"])
    work.loc[high, "target_pct"] = case["high_target"]
    daily_sum = work.groupby("buy_date")["target_pct"].transform("sum")
    work["target_pct"] = work["target_pct"] * (case["cap"] / daily_sum).clip(upper=1.0)
    ret_col = f"ret_h{case['hold']}"
    work = work.dropna(subset=[ret_col])
    if work.empty:
        return None
    work["weighted_ret"] = work["target_pct"] * work[ret_col]
    daily = work.groupby("buy_date", as_index=False).agg(
        daily_ret=("weighted_ret", "sum"),
        target_sum=("target_pct", "sum"),
        names=("stock_code", "count"),
    )
    daily["year"] = daily["buy_date"].str.slice(0, 4)
    total_ret = float(daily["daily_ret"].sum())
    sharpe_proxy = float(daily["daily_ret"].mean() / daily["daily_ret"].std() * (244 ** 0.5)) if daily["daily_ret"].std() else 0.0
    by_year = daily.groupby("year")["daily_ret"].sum().to_dict()
    min_year = min(float(v) for v in by_year.values()) if by_year else 0.0
    max_year_share = max(abs(float(v)) for v in by_year.values()) / max(abs(total_ret), 1e-9)
    monthly = daily.assign(month=daily["buy_date"].str.slice(0, 6)).groupby("month")["daily_ret"].sum()
    top_month_share = float(monthly.max() / max(total_ret, 1e-9)) if len(monthly) else 1.0
    recent60 = float(daily.tail(60)["daily_ret"].sum()) if len(daily) >= 60 else float(daily["daily_ret"].sum())
    objective = (
        total_ret
        + 1.2 * sharpe_proxy
        + 2.0 * min_year
        + 1.0 * recent60
        - 2.5 * max(0.0, max_year_share - 0.45)
        - 2.0 * max(0.0, top_month_share - 0.20)
    )
    return {
        "case": case["case"],
        "rows": int(len(work)),
        "buy_days": int(daily["buy_date"].nunique()),
        "stock_count": int(work["stock_code"].nunique()),
        "avg_names": float(daily["names"].mean()),
        "mean_target_sum": float(daily["target_sum"].mean()),
        "max_target_sum": float(daily["target_sum"].max()),
        "proxy_total_ret": total_ret,
        "proxy_sharpe": sharpe_proxy,
        "proxy_recent60": recent60,
        "proxy_min_year": min_year,
        "proxy_max_year_share": max_year_share,
        "proxy_top_month_share": top_month_share,
        "objective": objective,
        **{k: v for k, v in case.items() if k != "weights"},
        "w10": w[0],
        "w5": w[1],
        "w3": w[2],
        "w1": w[3],
    }


def make_cases() -> list[dict]:
    cases = []
    weights = [
        (0.60, 0.20, 0.00, 0.20),
        (0.50, 0.20, 0.10, 0.20),
        (0.70, 0.15, 0.00, 0.15),
    ]
    grids = itertools.product(
        [3, 4],
        [0.94, 0.96],
        [0.0, 0.60],
        [-8.0, -4.0],
        [0.5, 1.5],
        [120000, 250000],
        [200000],
        [10.0],
        [1, 2],
        [0.0, 0.01],
    )
    idx = 0
    for top_n, p10, p1, pct_min, gap_max, amount, mv, atr, hold, liq_bonus in grids:
        for weight in weights:
            idx += 1
            cases.append(
                {
                    "case": f"mlsmooth_{idx:05d}",
                    "weights": weight,
                    "top_n": top_n,
                    "p10_min": p10,
                    "p5_min": 0.0,
                    "p1_min": p1,
                    "pct_min": pct_min,
                    "pct_max": -1.5,
                    "gap_min": -8.0,
                    "gap_max": gap_max,
                    "amount_min": amount,
                    "mv_min": mv,
                    "atr_max": atr,
                    "hold": hold,
                    "liq_bonus": liq_bonus,
                    "cap": 0.96,
                    "base_target": 0.32,
                    "high_score": 0.985,
                    "deep_pct": -5.0,
                    "high_target": 0.46,
                }
            )
    return cases


def main() -> None:
    candidates = load_candidates()
    returns = load_returns()
    df = candidates.merge(returns, on=["stock_code", "buy_date"], how="left")
    rows = []
    for case in make_cases():
        result = score_case(df, case)
        if result is not None:
            rows.append(result)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("objective", ascending=False)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(frame.head(200).to_dict("records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "rows": int(len(frame)), "top": frame.head(10).to_dict("records")}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
