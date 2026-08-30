from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[2]
MAIN=ROOT/"quant"/"main"
RUNNER=MAIN/"run_juejin_signal_backtest.py"
STRATEGY_DIR=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_robust_rules_20260719"/"research_code_snapshot"
SOURCE_DIR=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_multihorizon_refine_juejin_20260720"
SIGNAL_FILE=SOURCE_DIR/"signals"/"w10_100_plain_mh65_pct3_m98_top1_h12_full100.csv"
SCORE_DB=SOURCE_DIR/"score_assets"/"w10_100.duckdb"
MARKET_DB=ROOT/"quant"/"data_file"/"production_assets"/"duckdb"/"l2_stock_daily_data.duckdb"
REPORT_DIR=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_repeat_topup_juejin_20260720"
LOG_DIR=REPORT_DIR/"logs"


def parse(path:Path)->dict:
    text=path.read_text(encoding="utf-8",errors="ignore") if path.exists() else ""
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line: continue
        payload=line.split("GM_BACKTEST_INDICATOR:",1)[1].strip()
        try:return ast.literal_eval(payload)
        except Exception:
            out={}
            for key in ["pnl_ratio","pnl_ratio_annual","sharp_ratio","max_drawdown","win_ratio"]:
                m=re.search(rf"'{key}':\s*([-+0-9.eE]+)",payload)
                if m:out[key]=float(m.group(1))
            for key in ["open_count","close_count"]:
                m=re.search(rf"'{key}':\s*([0-9]+)",payload)
                if m:out[key]=int(m.group(1))
            return out
    return {}


def run(mult:float)->dict:
    name=f"repeat_top1_topup_x{str(mult).replace('.','p')}"
    log=LOG_DIR/f"{name}.log"
    env=os.environ.copy();env.update({
        "GM_INTRADAY_RISK_MODE":"0","GM_ADAPTIVE_LIQUIDITY_SLIPPAGE":"0","GM_OPEN_DAILY_SCORE_EXIT":"0",
        "GM_INDEPENDENT_REPLACE_MODE":"0","GM_MAX_DAILY_BUYS":"1","GM_RESIZE_HELD_ON_SIGNAL":"1",
        "GM_RESIZE_HELD_TARGET_MULT":str(mult),"GM_RESIZE_HELD_MIN_DELTA_PCT":"0.005",
    })
    command=[sys.executable,str(RUNNER),"--strategy-dir",str(STRATEGY_DIR),"--signal-file",str(SIGNAL_FILE),"--log-file",str(log),
             "--max-positions","12","--holding-days","12","--max-holding-days","12","--target-position-pct","0.25",
             "--score-db",str(SCORE_DB),"--score-table","blended_rank_score","--market-db",str(MARKET_DB),
             "--backtest-adjust","none","--backtest-slippage-ratio","0.003","--backtest-start","2022-06-07 09:00:00","--backtest-end","2026-07-17 15:30:00"]
    proc=subprocess.run(command,cwd=str(MAIN),env=env,capture_output=True,text=True);ind=parse(log)
    return {"name":name,"topup_mult":mult,"returncode":proc.returncode,"annual_return":ind.get("pnl_ratio_annual"),"sharpe":ind.get("sharp_ratio"),"max_drawdown":ind.get("max_drawdown"),"win_ratio":ind.get("win_ratio"),"open_count":ind.get("open_count"),"close_count":ind.get("close_count"),"signal_file":str(SIGNAL_FILE),"log_file":str(log)}


def main()->None:
    REPORT_DIR.mkdir(parents=True,exist_ok=True);LOG_DIR.mkdir(exist_ok=True);rows=[]
    for mult in [1.25,1.5,2.0,2.5,3.0]:
        result=run(mult);rows.append(result);pd.DataFrame(rows).sort_values(["sharpe","annual_return"],ascending=False).to_csv(REPORT_DIR/"juejin_results.csv",index=False,encoding="utf-8-sig")
        print(json.dumps({k:result.get(k) for k in ["name","returncode","annual_return","sharpe","max_drawdown"]},ensure_ascii=False),flush=True)
    (REPORT_DIR/"result.json").write_text(json.dumps({"status":"research_only","results":rows},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
