from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DB = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_persistent_edge_20260720" / "persistent_edge_features.duckdb"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_entry_timing_20260720"

SCORES = {
    "plain": "rank_10d",
    "r1b02": "rank_10d + 0.02 * rank_1d",
    "r1b05": "rank_10d + 0.05 * rank_1d",
    "r1b10": "rank_10d + 0.10 * rank_1d",
    "r3b05": "rank_10d + 0.05 * rank_3d",
    "r5b05": "rank_10d + 0.05 * rank_5d",
    "agree10": "rank_10d - 0.10 * spread_3510",
    "improve05": "rank_10d + 0.05 * greatest(least(d10_1, 0.20), -0.20)",
}

FILTERS = {
    "base": "TRUE",
    "r1ge50": "rank_1d >= 0.50",
    "r1ge70": "rank_1d >= 0.70",
    "r1ge85": "rank_1d >= 0.85",
    "d10ge_m10": "d10_1 >= -0.10",
    "d10ge_m05": "d10_1 >= -0.05",
    "persist80": "rank10_lag1 >= 0.80",
    "persist90": "rank10_lag1 >= 0.90",
}


def stats(daily: pd.Series) -> tuple[float, float, float]:
    equity = (1 + daily).cumprod()
    annual = float(equity.iloc[-1] ** (252 / len(daily)) - 1) if len(daily) and equity.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=0)) if len(daily) else 0.0
    sharpe = float(daily.mean() / std * math.sqrt(252)) if std else 0.0
    mdd = float(abs((equity / equity.cummax() - 1).min())) if len(daily) else 0.0
    return annual, sharpe, mdd


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    rows = []
    try:
        all_dates = pd.Index(con.execute("SELECT DISTINCT trade_date FROM persistent_edge_features ORDER BY trade_date").fetchnumpy()["trade_date"])
        for score_name, score_expr in SCORES.items():
            for filter_name, filter_expr in FILTERS.items():
                selected = con.execute(
                    f"""
                    WITH ranked AS (
                        SELECT *, row_number() OVER (
                            PARTITION BY trade_date ORDER BY {score_expr} DESC, stock_code
                        ) AS pick_rank
                        FROM persistent_edge_features
                        WHERE rank_10d >= 0.98
                          AND least(rank_3d, rank_5d, rank_10d) >= 0.65
                          AND signal_pct_chg_raw <= 3.0
                          AND {filter_expr}
                    )
                    SELECT * FROM ranked WHERE pick_rank <= 2
                    """
                ).fetchdf()
                for topn in [1, 2]:
                    for hold in [10, 12, 15]:
                        ret_col = f"ret_open_{hold}"
                        work = selected[(selected["pick_rank"] <= topn) & selected[ret_col].notna()].copy()
                        work["net"] = work[ret_col] - 0.006
                        daily = (work.groupby("trade_date")["net"].mean() / hold).reindex(all_dates, fill_value=0.0)
                        full = stats(daily)
                        early = stats(daily[daily.index < "20250101"])
                        late = stats(daily[daily.index >= "20250101"])
                        dated = daily.copy(); dated.index = pd.to_datetime(dated.index)
                        years = [stats(group) for _, group in dated.groupby(dated.index.year) if len(group) >= 80]
                        worst_a = min((x[0] for x in years), default=-1.0)
                        worst_s = min((x[1] for x in years), default=-9.0)
                        rows.append({
                            "name": f"{score_name}_{filter_name}_top{topn}_h{hold}",
                            "score": score_name, "filter": filter_name, "topn": topn, "hold": hold,
                            "rows": len(work), "days": work["trade_date"].nunique(),
                            "annual_proxy": full[0], "sharpe_proxy": full[1], "mdd_proxy": full[2],
                            "early_annual": early[0], "early_sharpe": early[1],
                            "late_annual": late[0], "late_sharpe": late[1],
                            "worst_year_annual": worst_a, "worst_year_sharpe": worst_s,
                            "robust_score": min(early[1], late[1], worst_s) + 0.2 * full[1],
                        })
    finally:
        con.close()
    frame = pd.DataFrame(rows).sort_values(["robust_score", "sharpe_proxy"], ascending=False)
    frame.to_csv(REPORT_DIR / "local_screen.csv", index=False, encoding="utf-8-sig")
    eligible = frame[(frame["days"] >= 700) & (frame["early_annual"] > 0) & (frame["late_annual"] > 0)]
    eligible.head(100).to_csv(REPORT_DIR / "eligible_top100.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "result.json").write_text(json.dumps({"status":"research_only","cases":len(frame),"top":eligible.head(20).to_dict(orient="records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(eligible.head(20)[["name","annual_proxy","sharpe_proxy","mdd_proxy","worst_year_annual"]].to_string(index=False))


if __name__ == "__main__":
    main()
