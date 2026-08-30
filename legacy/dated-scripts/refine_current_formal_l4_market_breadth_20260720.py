from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import pandas as pd


ROOT=Path(__file__).resolve().parents[2]
FEATURE_DB=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_persistent_edge_20260720"/"persistent_edge_features.duckdb"
REPORT_DIR=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_market_breadth_20260720"


def stats(daily:pd.Series)->tuple[float,float,float]:
    eq=(1+daily).cumprod(); annual=float(eq.iloc[-1]**(252/len(daily))-1) if len(daily) and eq.iloc[-1]>0 else -1.0
    std=float(daily.std(ddof=0)) if len(daily) else 0.0; sharpe=float(daily.mean()/std*math.sqrt(252)) if std else 0.0
    mdd=float(abs((eq/eq.cummax()-1).min())) if len(daily) else 0.0; return annual,sharpe,mdd


def main()->None:
    REPORT_DIR.mkdir(parents=True,exist_ok=True);con=duckdb.connect(str(FEATURE_DB),read_only=True);rows=[]
    regimes={
        "all":"TRUE",
        "breadth40":"breadth >= 0.40",
        "breadth45":"breadth >= 0.45",
        "breadth50":"breadth >= 0.50",
        "breadth5_40":"breadth5 >= 0.40",
        "breadth5_45":"breadth5 >= 0.45",
        "breadth5_50":"breadth5 >= 0.50",
        "breadth10_40":"breadth10 >= 0.40",
        "breadth10_45":"breadth10 >= 0.45",
        "breadth10_50":"breadth10 >= 0.50",
        "avg5_ge_m02":"avg_pct5 >= -0.2",
        "avg5_ge_0":"avg_pct5 >= 0.0",
        "avg10_ge_m02":"avg_pct10 >= -0.2",
        "avg10_ge_0":"avg_pct10 >= 0.0",
    }
    try:
        con.execute("""
            CREATE TEMP TABLE market_breadth AS
            WITH daily AS (
                SELECT trade_date,
                       avg(CASE WHEN signal_pct_chg_raw > 0 THEN 1.0 ELSE 0.0 END) AS breadth,
                       avg(signal_pct_chg_raw) AS avg_pct
                FROM persistent_edge_features GROUP BY trade_date
            )
            SELECT *, avg(breadth) OVER (ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS breadth5,
                      avg(breadth) OVER (ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS breadth10,
                      avg(avg_pct) OVER (ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS avg_pct5,
                      avg(avg_pct) OVER (ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS avg_pct10
            FROM daily
        """)
        all_dates=pd.Index(con.execute("SELECT trade_date FROM market_breadth ORDER BY trade_date").fetchnumpy()["trade_date"])
        for regime,condition in regimes.items():
            selected=con.execute(f"""
                WITH ranked AS (
                    SELECT f.*,b.breadth,b.breadth5,b.breadth10,b.avg_pct5,b.avg_pct10,
                           row_number() OVER (PARTITION BY f.trade_date ORDER BY f.rank_10d DESC,f.stock_code) pick_rank
                    FROM persistent_edge_features f JOIN market_breadth b USING(trade_date)
                    WHERE f.rank_10d >= 0.98 AND least(f.rank_3d,f.rank_5d,f.rank_10d)>=0.65
                      AND f.signal_pct_chg_raw<=3.0 AND {condition}
                ) SELECT * FROM ranked WHERE pick_rank=1
            """).fetchdf()
            for hold in [10,12,15]:
                col=f"ret_open_{hold}";work=selected[selected[col].notna()].copy();work["net"]=work[col]-0.006
                daily=(work.groupby("trade_date")["net"].mean()/hold).reindex(all_dates,fill_value=0.0)
                full=stats(daily);early=stats(daily[daily.index<"20250101"]);late=stats(daily[daily.index>="20250101"])
                dated=daily.copy();dated.index=pd.to_datetime(dated.index);years=[stats(g) for _,g in dated.groupby(dated.index.year) if len(g)>=80]
                worst_a=min((x[0] for x in years),default=-1.0);worst_s=min((x[1] for x in years),default=-9.0)
                rows.append({"name":f"{regime}_h{hold}","regime":regime,"hold":hold,"rows":len(work),"days":work.trade_date.nunique(),
                             "annual_proxy":full[0],"sharpe_proxy":full[1],"mdd_proxy":full[2],"early_annual":early[0],"late_annual":late[0],
                             "worst_year_annual":worst_a,"worst_year_sharpe":worst_s,"robust_score":min(early[1],late[1],worst_s)+0.2*full[1]})
    finally:con.close()
    frame=pd.DataFrame(rows).sort_values(["robust_score","sharpe_proxy"],ascending=False);frame.to_csv(REPORT_DIR/"local_screen.csv",index=False,encoding="utf-8-sig")
    (REPORT_DIR/"result.json").write_text(json.dumps({"status":"research_only","results":frame.head(30).to_dict(orient="records")},ensure_ascii=False,indent=2),encoding="utf-8")
    print(frame.head(20)[["name","annual_proxy","sharpe_proxy","mdd_proxy","worst_year_annual","days"]].to_string(index=False))


if __name__=="__main__":main()
