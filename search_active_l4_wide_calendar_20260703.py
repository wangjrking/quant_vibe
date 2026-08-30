from __future__ import annotations

from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


QUANT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = QUANT_ROOT / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
OUT_CSV = REPORT_DIR / "active_l4_wide_calendar_search_summary.csv"
VARIANT_DIR = REPORT_DIR / "active_l4_wide_calendar_variants"

SLIPPAGE = 0.0015
BASELINE_ANNUAL = 0.4853


def metrics(daily_returns: pd.Series) -> tuple[float, float, float, float]:
    arr = daily_returns.to_numpy(dtype=float)
    n = len(arr)
    equity = np.cumprod(1.0 + arr)
    total = equity[-1] - 1.0 if n else 0.0
    annual = (equity[-1] ** (252.0 / n) - 1.0) if n and equity[-1] > 0 else -1.0
    std = arr.std(ddof=1) if n > 1 else 0.0
    sharpe = (arr.mean() / std * np.sqrt(252.0)) if std > 0 else 0.0
    peak = np.maximum.accumulate(equity) if n else np.array([1.0])
    dd = equity / peak - 1.0 if n else np.array([0.0])
    max_dd = -float(dd.min()) if n else 0.0
    return annual, sharpe, max_dd, total


def select_top(df: pd.DataFrame, sort_mode: str, topn: int) -> pd.DataFrame:
    if sort_mode == "score":
        sort_cols = ["buy_date", "pred_prob", "buy_amount"]
        ascending = [True, False, False]
    elif sort_mode == "liq":
        sort_cols = ["buy_date", "buy_amount", "pred_prob"]
        ascending = [True, False, False]
    elif sort_mode == "score_liq":
        df = df.copy()
        df["sort_score"] = df["score_pct_rank"] + 0.04 * np.log1p(df["buy_amount"].clip(lower=0))
        sort_cols = ["buy_date", "sort_score", "pred_prob"]
        ascending = [True, False, False]
    elif sort_mode == "low_gap_score":
        df = df.copy()
        df["sort_score"] = df["score_pct_rank"] - 0.015 * df["buy_open_gap_raw_pct"].fillna(0)
        sort_cols = ["buy_date", "sort_score", "pred_prob"]
        ascending = [True, False, False]
    else:
        raise ValueError(sort_mode)
    return df.sort_values(sort_cols, ascending=ascending).groupby("buy_date", sort=False).head(topn)


def main() -> None:
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    cal = con.execute(
        """
        select distinct buy_date
        from active_l4_wide
        where buy_date between '20220607' and '20260702'
        order by buy_date
        """
    ).fetchdf()["buy_date"].astype(str)
    base = con.execute(
        """
        select
            signal_date, buy_date, stock_code, name, market,
            pred_prob, score_pct_rank, score_desc_rank,
            signal_pct_chg, signal_open_gap_raw_pct,
            signal_amount, signal_turnover_rate, signal_total_mv, signal_atr_qfq,
            buy_open_gap_raw_pct, buy_amount, buy_turnover_rate, buy_total_mv, buy_atr_qfq,
            ret_h1, ret_h2, ret_h3, ret_h5
        from active_l4_wide
        where score_pct_rank >= 0.80
          and signal_amount >= 50000
          and signal_total_mv >= 50000
          and buy_amount >= 50000
          and ret_h1 is not null
        """
    ).fetchdf()
    con.close()

    for c in ["signal_pct_chg", "signal_open_gap_raw_pct", "buy_open_gap_raw_pct"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    rows: list[dict] = []
    cases: list[tuple[str, pd.DataFrame, dict]] = []
    score_mins = [0.80, 0.88, 0.93, 0.97]
    pct_maxs = [None, -1.75, -3.0]
    pct_mins = [None, -12.0]
    sig_gap_maxs = [None]
    buy_gap_ranges = [(None, None), (-3.0, 1.5), (-1.0, 1.0)]
    amount_mins = [50000, 90000, 150000]
    mv_mins = [50000, 200000]
    topn_values = [1, 3, 5]
    hold_values = [1, 2, 3]
    sort_modes = ["score", "liq", "low_gap_score"]

    case_id = 0
    for score_min, pct_max, pct_min, sig_gap_max, buy_gap, amt_min, mv_min, topn, hold, sort_mode in product(
        score_mins,
        pct_maxs,
        pct_mins,
        sig_gap_maxs,
        buy_gap_ranges,
        amount_mins,
        mv_mins,
        topn_values,
        hold_values,
        sort_modes,
    ):
        df = base
        mask = (
            (df["score_pct_rank"] >= score_min)
            & (df["signal_amount"] >= amt_min)
            & (df["signal_total_mv"] >= mv_min)
        )
        if pct_max is not None:
            mask &= df["signal_pct_chg"] <= pct_max
        if pct_min is not None:
            mask &= df["signal_pct_chg"] >= pct_min
        if sig_gap_max is not None:
            mask &= df["signal_open_gap_raw_pct"] <= sig_gap_max
        bg_min, bg_max = buy_gap
        if bg_min is not None:
            mask &= df["buy_open_gap_raw_pct"] >= bg_min
        if bg_max is not None:
            mask &= df["buy_open_gap_raw_pct"] <= bg_max
        cand = df.loc[mask]
        if len(cand) < 30:
            continue
        picked = select_top(cand, sort_mode, topn)
        if picked["buy_date"].nunique() < 30:
            continue
        ret_col = f"ret_h{hold}"
        picked = picked[picked[ret_col].notna()]
        if picked.empty:
            continue
        target_each = min(0.99 / topn, 0.60)
        adjusted_ret = ((1.0 + picked[ret_col]) * (1.0 - SLIPPAGE) / (1.0 + SLIPPAGE) - 1.0)
        daily = (adjusted_ret * target_each).groupby(picked["buy_date"].astype(str)).sum()
        daily = daily.reindex(cal, fill_value=0.0)
        annual, sharpe, max_dd, total = metrics(daily)
        n_open = int(picked["buy_date"].nunique())
        row = {
            "case_id": case_id,
            "annual": annual,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "total_return": total,
            "n_rows": int(len(picked)),
            "n_open": n_open,
            "score_min": score_min,
            "pct_min": pct_min,
            "pct_max": pct_max,
            "sig_gap_max": sig_gap_max,
            "buy_gap_min": bg_min,
            "buy_gap_max": bg_max,
            "amount_min": amt_min,
            "mv_min": mv_min,
            "topn": topn,
            "hold": hold,
            "sort_mode": sort_mode,
            "target_each": target_each,
        }
        rows.append(row)
        if annual > BASELINE_ANNUAL and max_dd < 0.40:
            cases.append((f"widecal_{case_id:05d}", picked.copy(), row))
        case_id += 1

    out = pd.DataFrame(rows).sort_values(["annual", "sharpe"], ascending=False)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    manifest_rows = []
    for name, picked, row in sorted(cases, key=lambda x: x[2]["annual"], reverse=True)[:12]:
        path = VARIANT_DIR / f"{name}.csv"
        hold = int(row["hold"])
        topn = int(row["topn"])
        target_each = float(row["target_each"])
        sig = picked.sort_values(["buy_date", "pred_prob"], ascending=[True, False]).copy()
        sig["target_weight"] = target_each
        sig["signal_score"] = sig["pred_prob"]
        sig["hold_days"] = hold
        sig["rank"] = sig.groupby("buy_date").cumcount() + 1
        sig[[
            "signal_date", "buy_date", "stock_code", "name", "market", "rank",
            "target_weight", "signal_score", "pred_prob", "score_pct_rank",
            "signal_pct_chg", "signal_open_gap_raw_pct", "buy_open_gap_raw_pct",
            "signal_amount", "buy_amount", "signal_total_mv", "buy_total_mv",
            "hold_days",
        ]].to_csv(path, index=False, encoding="utf-8-sig")
        mr = dict(row)
        mr["name"] = name
        mr["signal_file"] = str(path)
        manifest_rows.append(mr)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = REPORT_DIR / "active_l4_wide_calendar_variant_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(out.head(30).to_string(index=False))
    print("summary", OUT_CSV)
    print("manifest", manifest_path, "rows", len(manifest))


if __name__ == "__main__":
    main()
