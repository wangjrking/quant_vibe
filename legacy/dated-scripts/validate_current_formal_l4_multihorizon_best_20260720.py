from __future__ import annotations

import ast,json,os,re,subprocess,sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];MAIN=ROOT/"quant"/"main";RUNNER=MAIN/"run_juejin_signal_backtest.py"
STRATEGY_DIR=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_robust_rules_20260719"/"research_code_snapshot"
SOURCE=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_multihorizon_refine_juejin_20260720"
SIGNAL=SOURCE/"signals"/"w10_100_plain_mh65_pct3_m98_top1_h12_full100.csv";SCORE=SOURCE/"score_assets"/"w10_100.duckdb"
MARKET=ROOT/"quant"/"data_file"/"production_assets"/"duckdb"/"l2_stock_daily_data.duckdb"
REPORT=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_multihorizon_best_validation_20260720";LOGS=REPORT/"logs"
PERIODS=[("full_repeat","2022-06-07 09:00:00","2026-07-17 15:30:00"),("2022h2","2022-06-07 09:00:00","2022-12-30 15:30:00"),("2023","2023-01-03 09:00:00","2023-12-29 15:30:00"),("2024","2024-01-02 09:00:00","2024-12-31 15:30:00"),("2025","2025-01-02 09:00:00","2025-12-31 15:30:00"),("2026ytd","2026-01-05 09:00:00","2026-07-17 15:30:00"),("recent120","2026-01-20 09:00:00","2026-07-17 15:30:00"),("recent60","2026-04-20 09:00:00","2026-07-17 15:30:00")]

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

def run(name,start,end):
    log=LOGS/f"{name}.log";env=os.environ.copy();env.update({"GM_INTRADAY_RISK_MODE":"0","GM_ADAPTIVE_LIQUIDITY_SLIPPAGE":"0","GM_OPEN_DAILY_SCORE_EXIT":"0","GM_INDEPENDENT_REPLACE_MODE":"0","GM_MAX_DAILY_BUYS":"0","GM_RESIZE_HELD_ON_SIGNAL":"0"})
    cmd=[sys.executable,str(RUNNER),"--strategy-dir",str(STRATEGY_DIR),"--signal-file",str(SIGNAL),"--log-file",str(log),"--max-positions","12","--holding-days","12","--max-holding-days","12","--target-position-pct",str(1/12),"--score-db",str(SCORE),"--score-table","blended_rank_score","--market-db",str(MARKET),"--backtest-adjust","none","--backtest-slippage-ratio","0.003","--backtest-start",start,"--backtest-end",end]
    p=subprocess.run(cmd,cwd=str(MAIN),env=env,capture_output=True,text=True);i=parse(log)
    return {"name":name,"start":start,"end":end,"returncode":p.returncode,"annual_return":i.get("pnl_ratio_annual"),"cumulative_return":i.get("pnl_ratio"),"sharpe":i.get("sharp_ratio"),"max_drawdown":i.get("max_drawdown"),"win_ratio":i.get("win_ratio"),"open_count":i.get("open_count"),"close_count":i.get("close_count"),"log_file":str(log)}

def main():
    REPORT.mkdir(parents=True,exist_ok=True);LOGS.mkdir(exist_ok=True);rows=[]
    for period in PERIODS:
        x=run(*period);rows.append(x);pd.DataFrame(rows).to_csv(REPORT/"juejin_period_results.csv",index=False,encoding="utf-8-sig");print(json.dumps({k:x.get(k) for k in ["name","returncode","annual_return","sharpe","max_drawdown"]},ensure_ascii=False),flush=True)
    frame=pd.read_csv(SIGNAL,dtype=str);audit={"status":"research_only","signal_file":str(SIGNAL),"signal_rows":len(frame),"signal_days":frame.signal_date.nunique(),"min_signal_date":frame.signal_date.min(),"max_signal_date":frame.signal_date.max(),"min_buy_date":frame.buy_date.min(),"max_buy_date":frame.buy_date.max(),"duplicate_signal_symbol":int(frame.duplicated(["signal_date","stock_code"]).sum()),"bj_rows":int(frame.stock_code.str.endswith(".BJ").sum()),"target_pct_min":float(pd.to_numeric(frame.target_pct).min()),"target_pct_max":float(pd.to_numeric(frame.target_pct).max()),"results":rows}
    (REPORT/"validation.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":main()
