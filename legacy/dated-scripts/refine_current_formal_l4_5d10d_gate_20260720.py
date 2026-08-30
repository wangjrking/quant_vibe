from __future__ import annotations
import json,math
from pathlib import Path
import duckdb,pandas as pd
ROOT=Path(__file__).resolve().parents[2];DB=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_persistent_edge_20260720"/"persistent_edge_features.duckdb";OUT=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_5d10d_gate_20260720"
def stats(x):
 e=(1+x).cumprod();a=float(e.iloc[-1]**(252/len(x))-1) if len(x) and e.iloc[-1]>0 else -1.;s=float(x.std(ddof=0)) if len(x) else 0.;q=float(x.mean()/s*math.sqrt(252)) if s else 0.;d=float(abs((e/e.cummax()-1).min())) if len(x) else 0.;return a,q,d
def main():
 OUT.mkdir(parents=True,exist_ok=True);c=duckdb.connect(str(DB),read_only=True);rows=[]
 try:
  dates=pd.Index(c.execute("select distinct trade_date from persistent_edge_features order by trade_date").fetchnumpy()["trade_date"])
  for floor in [.4,.5,.6,.7,.8,.9]:
   for pct in [1.,2.,3.]:
    f=c.execute(f"""with r as(select *,row_number() over(partition by trade_date order by rank_10d desc,stock_code) pick_rank from persistent_edge_features where least(rank_5d,rank_10d)>={floor} and signal_pct_chg_raw<={pct}) select * from r where pick_rank<=2""").fetchdf()
    for th in [.90,.95,.98]:
     for topn in [1,2]:
      for h in [10,12,15]:
       col=f"ret_open_{h}";w=f[(f.rank_10d>=th)&(f.pick_rank<=topn)&f[col].notna()].copy();w["net"]=w[col]-.006;day=(w.groupby("trade_date").net.mean()/h).reindex(dates,fill_value=0);full=stats(day);early=stats(day[day.index<"20250101"]);late=stats(day[day.index>="20250101"]);z=day.copy();z.index=pd.to_datetime(z.index);ys=[stats(g) for _,g in z.groupby(z.index.year) if len(g)>=80];wa=min((x[0] for x in ys),default=-1);ws=min((x[1] for x in ys),default=-9)
       rows.append({"name":f"f{int(floor*100)}_pct{int(pct)}_m{int(th*100)}_top{topn}_h{h}","floor":floor,"pct":pct,"threshold":th,"topn":topn,"hold":h,"rows":len(w),"days":w.trade_date.nunique(),"annual_proxy":full[0],"sharpe_proxy":full[1],"mdd_proxy":full[2],"early_annual":early[0],"late_annual":late[0],"worst_year_annual":wa,"worst_year_sharpe":ws,"robust_score":min(early[1],late[1],ws)+.2*full[1]})
 finally:c.close()
 x=pd.DataFrame(rows).sort_values(["robust_score","sharpe_proxy"],ascending=False);x.to_csv(OUT/"local_screen.csv",index=False,encoding="utf-8-sig");(OUT/"result.json").write_text(json.dumps({"status":"research_only","top":x.head(30).to_dict(orient="records")},ensure_ascii=False,indent=2),encoding="utf-8");print(x.head(20)[["name","annual_proxy","sharpe_proxy","mdd_proxy","worst_year_annual","days"]].to_string(index=False))
if __name__=="__main__":main()
