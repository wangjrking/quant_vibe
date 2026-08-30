from __future__ import annotations

import ast,json,os,re,subprocess,sys
from pathlib import Path
import duckdb
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];MAIN=ROOT/"quant"/"main";RUNNER=MAIN/"run_juejin_signal_backtest.py"
STRATEGY_DIR=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_robust_rules_20260719"/"research_code_snapshot"
SOURCE=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_multihorizon_refine_juejin_20260720"
BASE_SIGNAL=SOURCE/"signals"/"w10_100_plain_mh65_pct3_m98_top1_h12_full100.csv";SCORE_DB=SOURCE/"score_assets"/"w10_100.duckdb"
FEATURE_DB=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_persistent_edge_20260720"/"persistent_edge_features.duckdb"
MARKET_DB=ROOT/"quant"/"data_file"/"production_assets"/"duckdb"/"l2_stock_daily_data.duckdb"
REPORT_DIR=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_market_breadth_juejin_20260720";SIGNAL_DIR=REPORT_DIR/"signals";LOG_DIR=REPORT_DIR/"logs"
CASES=[("avg10_ge0_h12","avg_pct10 >= 0",12),("avg10_ge0_h15","avg_pct10 >= 0",15),("avg10_ge_m02_h15","avg_pct10 >= -0.2",15),("breadth10_45_h15","breadth10 >= 0.45",15)]

def parse(path):
    text=path.read_text(encoding="utf-8",errors="ignore") if path.exists() else ""
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:continue
        payload=line.split("GM_BACKTEST_INDICATOR:",1)[1].strip()
        try:return ast.literal_eval(payload)
        except Exception:
            out={}
            for k in ["pnl_ratio","pnl_ratio_annual","sharp_ratio","max_drawdown","win_ratio"]:
                m=re.search(rf"'{k}':\s*([-+0-9.eE]+)",payload)
                if m:out[k]=float(m.group(1))
            for k in ["open_count","close_count"]:
                m=re.search(rf"'{k}':\s*([0-9]+)",payload)
                if m:out[k]=int(m.group(1))
            return out
    return {}

def eligible_dates(condition):
    con=duckdb.connect(str(FEATURE_DB),read_only=True)
    try:return set(con.execute(f"""WITH daily AS (SELECT trade_date,avg(CASE WHEN signal_pct_chg_raw>0 THEN 1.0 ELSE 0.0 END) breadth,avg(signal_pct_chg_raw) avg_pct FROM persistent_edge_features GROUP BY trade_date),r AS (SELECT *,avg(breadth) OVER(ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) breadth10,avg(avg_pct) OVER(ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) avg_pct10 FROM daily) SELECT trade_date FROM r WHERE {condition}""").fetchnumpy()["trade_date"])
    finally:con.close()

def run(name,condition,hold):
    dates=eligible_dates(condition);frame=pd.read_csv(BASE_SIGNAL,dtype=str);frame=frame[frame.signal_date.isin(dates)].copy();frame["holding_days"]=hold;frame["max_holding_days"]=hold;frame["target_pct"]=1/hold
    signal=SIGNAL_DIR/f"{name}.csv";frame.to_csv(signal,index=False,encoding="utf-8-sig");log=LOG_DIR/f"{name}.log"
    env=os.environ.copy();env.update({"GM_INTRADAY_RISK_MODE":"0","GM_ADAPTIVE_LIQUIDITY_SLIPPAGE":"0","GM_OPEN_DAILY_SCORE_EXIT":"0","GM_INDEPENDENT_REPLACE_MODE":"0","GM_MAX_DAILY_BUYS":"1","GM_RESIZE_HELD_ON_SIGNAL":"0"})
    cmd=[sys.executable,str(RUNNER),"--strategy-dir",str(STRATEGY_DIR),"--signal-file",str(signal),"--log-file",str(log),"--max-positions",str(hold),"--holding-days",str(hold),"--max-holding-days",str(hold),"--target-position-pct",str(1/hold),"--score-db",str(SCORE_DB),"--score-table","blended_rank_score","--market-db",str(MARKET_DB),"--backtest-adjust","none","--backtest-slippage-ratio","0.003","--backtest-start","2022-06-07 09:00:00","--backtest-end","2026-07-17 15:30:00"]
    proc=subprocess.run(cmd,cwd=str(MAIN),env=env,capture_output=True,text=True);ind=parse(log)
    return {"name":name,"condition":condition,"hold":hold,"signal_days":frame.signal_date.nunique(),"returncode":proc.returncode,"annual_return":ind.get("pnl_ratio_annual"),"sharpe":ind.get("sharp_ratio"),"max_drawdown":ind.get("max_drawdown"),"win_ratio":ind.get("win_ratio"),"open_count":ind.get("open_count"),"close_count":ind.get("close_count"),"signal_file":str(signal),"log_file":str(log)}

def main():
    for p in [REPORT_DIR,SIGNAL_DIR,LOG_DIR]:p.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in CASES:
        x=run(*case);rows.append(x);pd.DataFrame(rows).sort_values(["sharpe","annual_return"],ascending=False).to_csv(REPORT_DIR/"juejin_results.csv",index=False,encoding="utf-8-sig");print(json.dumps({k:x.get(k) for k in ["name","returncode","annual_return","sharpe","max_drawdown"]},ensure_ascii=False),flush=True)
    (REPORT_DIR/"result.json").write_text(json.dumps({"status":"research_only","results":rows},ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":main()
