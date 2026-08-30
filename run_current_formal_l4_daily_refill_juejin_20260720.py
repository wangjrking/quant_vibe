from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SOURCE = ROOT / "quant" / "main" / "run_current_formal_l4_persistent_edge_juejin_20260720.py"
SPEC = importlib.util.spec_from_file_location("persistent_edge_runner", BASE_SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_daily_refill_juejin_20260720"
SIGNAL_DIR = REPORT_DIR / "signals"
SCORE_DIR = REPORT_DIR / "score_assets"
LOG_DIR = REPORT_DIR / "logs"
STRATEGY_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_robust_rules_20260719" / "research_code_snapshot"
RUNNER = ROOT / "quant" / "main" / "run_juejin_signal_backtest.py"

for floor in [0.60, 0.65, 0.70]:
    MODULE.FILTERS[f"mh{int(floor*100)}_pct3"] = f"least(rank_3d, rank_5d, rank_10d) >= {floor} AND signal_pct_chg_raw <= 3.0"

CASES = [
    {"floor":0.65,"depth":depth} for depth in [1,3,5,10]
] + [
    {"floor":floor,"depth":5} for floor in [0.60,0.70]
]


def parse_indicator(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            result = {}
            for key in ["pnl_ratio","pnl_ratio_annual","sharp_ratio","max_drawdown","win_ratio"]:
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match: result[key] = float(match.group(1))
            for key in ["open_count","close_count"]:
                match = re.search(rf"'{key}':\s*([0-9]+)", payload)
                if match: result[key] = int(match.group(1))
            return result
    return {}


def run(case: dict, con: duckdb.DuckDBPyConnection, score_db: Path) -> dict:
    floor = float(case["floor"]); depth = int(case["depth"])
    base_case = {"blend":"w10_100","score":"plain","filter":f"mh{int(floor*100)}_pct3","threshold":0.98,"topn":depth,"hold":12}
    name, generated, _max_positions, _target = MODULE.build_signal(con, base_case)
    frame = pd.read_csv(generated, dtype={"signal_date":str,"buy_date":str,"stock_code":str})
    frame["target_pct"] = 1.0 / 12.0
    name = f"mh{int(floor*100)}_m98_h12_refill_depth{depth}_daily1"
    signal_file = SIGNAL_DIR / f"{name}.csv"
    frame.to_csv(signal_file, index=False, encoding="utf-8-sig")
    log_file = LOG_DIR / f"{name}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE":"0","GM_ADAPTIVE_LIQUIDITY_SLIPPAGE":"0","GM_OPEN_DAILY_SCORE_EXIT":"0",
        "GM_INDEPENDENT_REPLACE_MODE":"0","GM_MAX_DAILY_BUYS":"1",
    })
    command = [
        sys.executable,str(RUNNER),"--strategy-dir",str(STRATEGY_DIR),"--signal-file",str(signal_file),"--log-file",str(log_file),
        "--max-positions","12","--holding-days","12","--max-holding-days","12","--target-position-pct",str(1/12),
        "--score-db",str(score_db),"--score-table","blended_rank_score","--market-db",str(MODULE.MARKET_DB),
        "--backtest-adjust","none","--backtest-slippage-ratio","0.003","--backtest-start","2022-06-07 09:00:00","--backtest-end","2026-07-17 15:30:00",
    ]
    process = subprocess.run(command,cwd=str(MODULE.MAIN),env=env,capture_output=True,text=True)
    indicator = parse_indicator(log_file)
    return {"name":name,**case,"signal_rows":len(frame),"signal_days":frame.signal_date.nunique(),"returncode":process.returncode,
            "annual_return":indicator.get("pnl_ratio_annual"),"sharpe":indicator.get("sharp_ratio"),"max_drawdown":indicator.get("max_drawdown"),
            "win_ratio":indicator.get("win_ratio"),"open_count":indicator.get("open_count"),"close_count":indicator.get("close_count"),
            "signal_file":str(signal_file),"score_db":str(score_db),"log_file":str(log_file)}


def main() -> None:
    for path in [REPORT_DIR,SIGNAL_DIR,SCORE_DIR,LOG_DIR]: path.mkdir(parents=True,exist_ok=True)
    MODULE.SIGNAL_DIR = SIGNAL_DIR
    MODULE.SCORE_DIR = SCORE_DIR
    con = duckdb.connect(str(MODULE.FEATURE_DB),read_only=True)
    rows=[]
    try:
        score_db=MODULE.build_score(con,"w10_100")
        for case in CASES:
            result=run(case,con,score_db); rows.append(result)
            pd.DataFrame(rows).sort_values(["sharpe","annual_return"],ascending=False,na_position="last").to_csv(REPORT_DIR/"juejin_results.csv",index=False,encoding="utf-8-sig")
            print(json.dumps({key:result.get(key) for key in ["name","returncode","annual_return","sharpe","max_drawdown","open_count"]},ensure_ascii=False),flush=True)
    finally: con.close()
    (REPORT_DIR/"result.json").write_text(json.dumps({"status":"research_only","results":rows},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
