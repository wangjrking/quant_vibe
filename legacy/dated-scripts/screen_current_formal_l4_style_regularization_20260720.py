from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DB = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_persistent_edge_20260720" / "persistent_edge_features.duckdb"
OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_style_regularization_20260720"


def stats(daily: pd.Series) -> tuple[float, float, float]:
    if daily.empty:
        return 0.0, 0.0, 0.0
    equity = (1.0 + daily).cumprod()
    annual = float(equity.iloc[-1] ** (252.0 / len(daily)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=0))
    sharpe = float(daily.mean() / std * math.sqrt(252.0)) if std else 0.0
    mdd = float(abs((equity / equity.cummax() - 1.0).min()))
    return annual, sharpe, mdd


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    rows = []
    try:
        dates = pd.Index(con.execute("select distinct trade_date from persistent_edge_features order by trade_date").fetchnumpy()["trade_date"])
        for amount_min in (90000, 150000, 200000):
            for mv_min in (200000, 500000, 800000):
                base = con.execute(
                    f"""
                    select * from persistent_edge_features
                    where rank_10d >= 0.98 and rank_5d >= 0.50
                      and signal_pct_chg_raw <= 3.0
                      and amount >= {amount_min} and total_mv >= {mv_min}
                    """
                ).fetchdf()
                for amount_w in (0.0, 0.01, 0.02, 0.03):
                    for mv_w in (0.0, 0.01, 0.02):
                        for atr_w in (0.0, 0.01, 0.02):
                            work = base.copy()
                            work["select_score"] = (
                                work["rank_10d"] + amount_w * work["amount_rank"]
                                + mv_w * work["mv_rank"] - atr_w * work["atr_rank"]
                            )
                            work = work.sort_values(["trade_date", "select_score", "stock_code"], ascending=[True, False, True])
                            work = work.groupby("trade_date", as_index=False).head(1)
                            work = work[(work["rank_1d"] >= 0.88) & work["ret_open_12"].notna()].copy()
                            daily = ((work.set_index("trade_date")["ret_open_12"] - 0.006) / 3.0).reindex(dates, fill_value=0.0)
                            full = stats(daily)
                            early = stats(daily[daily.index < "20240101"])
                            middle = stats(daily[(daily.index >= "20240101") & (daily.index < "20250101")])
                            late = stats(daily[daily.index >= "20250101"])
                            rows.append({
                                "name": f"a{amount_min}_mv{mv_min}_aw{amount_w:.2f}_mw{mv_w:.2f}_atr{atr_w:.2f}",
                                "amount_min": amount_min, "mv_min": mv_min, "amount_w": amount_w,
                                "mv_w": mv_w, "atr_w": atr_w, "rows": int(len(work)),
                                "annual_proxy": full[0], "sharpe_proxy": full[1], "mdd_proxy": full[2],
                                "early_annual": early[0], "early_sharpe": early[1],
                                "middle_annual": middle[0], "middle_sharpe": middle[1],
                                "late_annual": late[0], "late_sharpe": late[1],
                                "robust_score": min(early[1], middle[1], late[1]) + 0.2 * full[1],
                            })
    finally:
        con.close()
    frame = pd.DataFrame(rows).sort_values(["robust_score", "sharpe_proxy", "annual_proxy"], ascending=False)
    frame.to_csv(OUT / "local_screen.csv", index=False, encoding="utf-8-sig")
    top = frame.head(30).replace({np.nan: None}).to_dict(orient="records")
    (OUT / "result.json").write_text(json.dumps({"status": "research_only_local_prescreen", "cases": len(frame), "top": top}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
