from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_persistent_edge_20260720"
    / "persistent_edge_features.duckdb"
)
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_multihorizon_refine_20260720"
)


def period_stats(daily: pd.Series) -> tuple[float, float, float]:
    if daily.empty:
        return 0.0, 0.0, 0.0
    equity = (1.0 + daily).cumprod()
    annual = float(equity.iloc[-1] ** (252.0 / len(daily)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=0))
    sharpe = float(daily.mean() / std * math.sqrt(252.0)) if std else 0.0
    mdd = float(abs((equity / equity.cummax() - 1.0).min()))
    return annual, sharpe, mdd


def evaluate(selected: pd.DataFrame, all_dates: pd.Index, params: dict) -> dict:
    hold = int(params["hold"])
    ret_col = f"ret_open_{hold}"
    work = selected[
        (selected["rank_10d"] >= params["threshold"])
        & (selected["pick_rank"] <= params["topn"])
        & selected[ret_col].notna()
    ].copy()
    if params["gap_min"] is not None:
        work = work[
            (work["buy_open_gap_raw_pct"] >= params["gap_min"])
            & (work["buy_open_gap_raw_pct"] <= params["gap_max"])
        ]
    work["net"] = work[ret_col] - 0.006
    daily = (work.groupby("trade_date")["net"].mean() / hold).reindex(all_dates, fill_value=0.0)
    early = daily[daily.index < "20250101"]
    late = daily[daily.index >= "20250101"]
    full = period_stats(daily)
    early_stats = period_stats(early)
    late_stats = period_stats(late)
    dated = daily.copy()
    dated.index = pd.to_datetime(dated.index)
    years = [period_stats(group) for _, group in dated.groupby(dated.index.year) if len(group) >= 80]
    worst_year_annual = min((x[0] for x in years), default=-1.0)
    worst_year_sharpe = min((x[1] for x in years), default=-9.0)
    name = (
        f"mh{int(params['floor']*100)}_pct{str(params['pct_cap']).replace('.', 'p')}"
        f"_m{int(params['threshold']*100)}_top{params['topn']}_h{hold}_{params['gap_name']}"
    )
    return {
        "name": name,
        **params,
        "rows": int(len(work)),
        "days": int(work["trade_date"].nunique()),
        "mean_gross": float(work[ret_col].mean()) if len(work) else None,
        "annual_proxy": full[0],
        "sharpe_proxy": full[1],
        "mdd_proxy": full[2],
        "early_annual": early_stats[0],
        "early_sharpe": early_stats[1],
        "late_annual": late_stats[0],
        "late_sharpe": late_stats[1],
        "worst_year_annual": worst_year_annual,
        "worst_year_sharpe": worst_year_sharpe,
        "robust_score": min(early_stats[1], late_stats[1], worst_year_sharpe) + 0.2 * full[1],
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    rows = []
    gaps = [
        ("nogap", None, None),
        ("gap5m_1p", -5.0, 1.0),
        ("gap3m_1p", -3.0, 1.0),
        ("gap5m_0p", -5.0, 0.0),
        ("gap2m_2p", -2.0, 2.0),
    ]
    try:
        all_dates = pd.Index(
            con.execute("SELECT DISTINCT trade_date FROM persistent_edge_features ORDER BY trade_date")
            .fetchnumpy()["trade_date"]
        )
        for floor in [0.60, 0.65, 0.70, 0.75, 0.80]:
            for pct_cap in [0.5, 1.0, 2.0, 3.0]:
                selected = con.execute(
                    f"""
                    WITH ranked AS (
                        SELECT *, row_number() OVER (
                            PARTITION BY trade_date ORDER BY rank_10d DESC, stock_code
                        ) AS pick_rank
                        FROM persistent_edge_features
                        WHERE least(rank_3d, rank_5d, rank_10d) >= {floor}
                          AND signal_pct_chg_raw <= {pct_cap}
                    )
                    SELECT * FROM ranked WHERE pick_rank <= 3
                    """
                ).fetchdf()
                for threshold in [0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98]:
                    for topn in [1, 2, 3]:
                        for hold in [8, 10, 12, 15]:
                            for gap_name, gap_min, gap_max in gaps:
                                params = {
                                    "floor": floor,
                                    "pct_cap": pct_cap,
                                    "threshold": threshold,
                                    "topn": topn,
                                    "hold": hold,
                                    "gap_name": gap_name,
                                    "gap_min": gap_min,
                                    "gap_max": gap_max,
                                }
                                rows.append(evaluate(selected, all_dates, params))
    finally:
        con.close()
    frame = pd.DataFrame(rows).sort_values(["robust_score", "sharpe_proxy"], ascending=False)
    frame.to_csv(REPORT_DIR / "local_screen.csv", index=False, encoding="utf-8-sig")
    eligible = frame[
        (frame["days"] >= 700)
        & (frame["early_annual"] > 0)
        & (frame["late_annual"] > 0)
        & (frame["worst_year_annual"] > -0.05)
    ]
    eligible.head(200).to_csv(REPORT_DIR / "eligible_top200.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "result.json").write_text(
        json.dumps(
            {
                "status": "research_only",
                "cases": len(frame),
                "eligible": len(eligible),
                "top": eligible.head(20).where(pd.notna(eligible.head(20)), None).to_dict(orient="records"),
                "selection_uses_future_data": False,
                "buy_open_gap_is_execution_time_gate": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(eligible.head(20)[["name", "annual_proxy", "sharpe_proxy", "mdd_proxy", "worst_year_annual"]].to_string(index=False))


if __name__ == "__main__":
    main()
