from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "old500_date_selection_bias"
WIDE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
OLD_SIGNAL = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "qfq_rerank_refill_candidates"
    / "reconstruct_500_from_p44_plus_1dge_m002"
    / "signals.csv"
)
LATEST_PROFILE = (
    REPORT_DIR
    / "latest_l4_old500_profile_candidates"
    / "signals"
    / "prof_rank16_pct175_gap15_top1"
    / "signals.csv"
)


def _daily_stats(df: pd.DataFrame, value_col: str = "ret_h1", target_col: str = "target_pct") -> dict:
    x = df.copy()
    x[target_col] = x[target_col].astype(float)
    x[value_col] = x[value_col].astype(float)
    day = x.groupby("signal_date").apply(lambda g: (g[value_col] * g[target_col]).sum()).sort_index()
    if day.empty:
        return {}
    nav = (1.0 + day).cumprod()
    peak = nav.cummax()
    std = day.std(ddof=1)
    return {
        "rows": int(len(x)),
        "days": int(day.size),
        "mean_daily": float(day.mean()),
        "annual_proxy": float((1.0 + day.mean()) ** 252 - 1.0),
        "sharpe_proxy": float(day.mean() / std * (252 ** 0.5)) if std else None,
        "mdd_proxy": float(-(nav / peak - 1.0).min()),
        "win_day_ratio": float((day > 0).mean()),
        "sum_weighted_ret": float(day.sum()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(OLD_SIGNAL, dtype={"signal_date": str, "stock_code": str})
    latest = pd.read_csv(LATEST_PROFILE, dtype={"signal_date": str, "stock_code": str})
    old_days = set(old["signal_date"].astype(str))
    latest["is_old_signal_day"] = latest["signal_date"].isin(old_days)

    con = duckdb.connect(str(WIDE_DB), read_only=True)
    try:
        con.register("old_sig", old)
        con.register("latest_profile", latest)
        old_with_ret = con.execute(
            """
            SELECT
                o.*,
                w.ret_h1,
                w.score_desc_rank,
                w.score_pct_rank,
                w.buy_open_gap_raw_pct,
                w.signal_pct_chg
            FROM old_sig o
            JOIN active_l4_wide w
              ON w.signal_date = o.signal_date AND w.stock_code = o.stock_code
            """
        ).fetchdf()
        latest_with_ret = con.execute(
            """
            SELECT
                l.*,
                w.ret_h1,
                w.score_desc_rank,
                w.score_pct_rank
            FROM latest_profile l
            JOIN active_l4_wide w
              ON w.signal_date = l.signal_date AND w.stock_code = l.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    old_with_ret.to_csv(OUT_DIR / "old500_with_current_ret_h1.csv", index=False, encoding="utf-8-sig")
    latest_with_ret.to_csv(OUT_DIR / "profile_top1_with_ret_h1.csv", index=False, encoding="utf-8-sig")

    latest_old_days = latest_with_ret[latest_with_ret["is_old_signal_day"]].copy()
    latest_non_old_days = latest_with_ret[~latest_with_ret["is_old_signal_day"]].copy()

    by_year_rows = []
    for label, df in [
        ("old500_actual", old_with_ret),
        ("profile_top1_all", latest_with_ret),
        ("profile_top1_old_days", latest_old_days),
        ("profile_top1_non_old_days", latest_non_old_days),
    ]:
        tmp = df.copy()
        tmp["year"] = tmp["signal_date"].str.slice(0, 4)
        for year, g in tmp.groupby("year"):
            s = _daily_stats(g)
            s["label"] = label
            s["year"] = year
            by_year_rows.append(s)
    by_year = pd.DataFrame(by_year_rows)
    by_year.to_csv(OUT_DIR / "date_selection_by_year.csv", index=False, encoding="utf-8-sig")

    summary = {
        "old500_actual": _daily_stats(old_with_ret),
        "profile_top1_all": _daily_stats(latest_with_ret),
        "profile_top1_old_days": _daily_stats(latest_old_days),
        "profile_top1_non_old_days": _daily_stats(latest_non_old_days),
        "old_signal_days": int(len(old_days)),
        "profile_total_days": int(latest_with_ret["signal_date"].nunique()),
        "profile_old_days_overlap": int(latest_old_days["signal_date"].nunique()),
        "profile_non_old_days": int(latest_non_old_days["signal_date"].nunique()),
    }
    (OUT_DIR / "date_selection_bias_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
