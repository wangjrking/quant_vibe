from __future__ import annotations
import ast,json,os,re,subprocess,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];MAIN=ROOT/"quant"/"main";RUNNER=MAIN/"run_juejin_signal_backtest.py"
STRATEGY=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_robust_rules_20260719"/"research_code_snapshot"
SRC=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_5d10d_gate_juejin_20260720";BASE=SRC/"signals"/"w10_100_plain_f50_pct3_m98_top1_h12_full100.csv";SCORE=SRC/"score_assets"/"w10_100.duckdb";MARKET=ROOT/"quant"/"data_file"/"production_assets"/"duckdb"/"l2_stock_daily_data.duckdb"
OUT=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_5d10d_dynamic_hold_juejin_20260720";SIG=OUT/"signals";LOG=OUT/"logs"
CASES={"r5_80_h15_else10":lambda x:15 if x.rank_5d>=.8 else 10,"r5_80_h15_else12":lambda x:15 if x.rank_5d>=.8 else 12,"r5_90_h15_else10":lambda x:15 if x.rank_5d>=.9 else 10,"r1_85_h10_else12":lambda x:10 if x.rank_1d>=.85 else 12,"r1_85_h12_else15":lambda x:12 if x.rank_1d>=.85 else 15}
def parse(p):
 t=p.read_text(encoding="utf-8",errors="ignore") if p.exists() else ""
 for l in reversed(t.splitlines()):
  if "GM_BACKTEST_INDICATOR:" not in l:continue
  q=l.split("GM_BACKTEST_INDICATOR:",1)[1].strip()
  try:return ast.literal_eval(q)
  except Exception:
   o={}
   for k in ["pnl_ratio","pnl_ratio_annual","sharp_ratio","max_drawdown","win_ratio"]:
    m=re.search(rf"'{k}':\s*([-+0-9.eE]+)",q)
    if m:o[k]=float(m.group(1))
   for k in ["open_count","close_count"]:
    m=re.search(rf"'{k}':\s*([0-9]+)",q)
    if m:o[k]=int(m.group(1))
   return o
 return {}
def run(name,fn):
 f=pd.read_csv(BASE);f["holding_days"]=f.apply(fn,axis=1);f["max_holding_days"]=f.holding_days;f["target_pct"]=1/12;s=SIG/f"{name}.csv";f.to_csv(s,index=False,encoding="utf-8-sig");log=LOG/f"{name}.log";e=os.environ.copy();e.update({"GM_INTRADAY_RISK_MODE":"0","GM_ADAPTIVE_LIQUIDITY_SLIPPAGE":"0","GM_OPEN_DAILY_SCORE_EXIT":"0","GM_INDEPENDENT_REPLACE_MODE":"0","GM_RESIZE_HELD_ON_SIGNAL":"0"});cmd=[sys.executable,str(RUNNER),"--strategy-dir",str(STRATEGY),"--signal-file",str(s),"--log-file",str(log),"--max-positions","12","--holding-days","12","--max-holding-days","15","--target-position-pct",str(1/12),"--score-db",str(SCORE),"--score-table","blended_rank_score","--market-db",str(MARKET),"--backtest-adjust","none","--backtest-slippage-ratio","0.003","--backtest-start","2022-06-07 09:00:00","--backtest-end","2026-07-17 15:30:00"];p=subprocess.run(cmd,cwd=str(MAIN),env=e,capture_output=True,text=True);i=parse(log);return {"name":name,"returncode":p.returncode,"annual_return":i.get("pnl_ratio_annual"),"sharpe":i.get("sharp_ratio"),"max_drawdown":i.get("max_drawdown"),"win_ratio":i.get("win_ratio"),"open_count":i.get("open_count"),"close_count":i.get("close_count"),"signal_file":str(s),"log_file":str(log)}
def main():
 [p.mkdir(parents=True,exist_ok=True) for p in [OUT,SIG,LOG]];rows=[]
 for n,f in CASES.items():
  x=run(n,f);rows.append(x);pd.DataFrame(rows).sort_values(["sharpe","annual_return"],ascending=False).to_csv(OUT/"juejin_results.csv",index=False,encoding="utf-8-sig");print(json.dumps({k:x.get(k) for k in ["name","returncode","annual_return","sharpe","max_drawdown"]},ensure_ascii=False),flush=True)
 (OUT/"result.json").write_text(json.dumps({"status":"research_only","results":rows},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
