from __future__ import annotations
import importlib.util,json
from pathlib import Path
import duckdb,pandas as pd

ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/"quant"/"main"/"run_current_formal_l4_persistent_edge_juejin_20260720.py"
SPEC=importlib.util.spec_from_file_location("runner",SOURCE);M=importlib.util.module_from_spec(SPEC);assert SPEC and SPEC.loader;SPEC.loader.exec_module(M)
REPORT=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_pure10_latest_juejin_20260720";M.REPORT_DIR=REPORT;M.SIGNAL_DIR=REPORT/"signals";M.SCORE_DIR=REPORT/"score_assets";M.LOG_DIR=REPORT/"logs"
M.FILTERS["pure10_broad"]="signal_pct_chg_raw <= 3.0"
CASES=[{"blend":"w10_100","score":"plain","filter":"pure10_broad","threshold":threshold,"topn":1,"hold":hold} for threshold in [0.90,0.95,0.98] for hold in [10,12,15]]

def main():
    for p in [REPORT,M.SIGNAL_DIR,M.SCORE_DIR,M.LOG_DIR]:p.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(M.FEATURE_DB),read_only=True);rows=[]
    try:
        score=M.build_score(con,"w10_100")
        for case in CASES:
            name,signal,maxpos,target=M.build_signal(con,case);x=M.run_case(case,name,signal,maxpos,target,score);rows.append(x)
            pd.DataFrame(rows).sort_values(["sharpe","annual_return"],ascending=False,na_position="last").to_csv(REPORT/"juejin_results.csv",index=False,encoding="utf-8-sig")
            print(json.dumps({k:x.get(k) for k in ["name","returncode","annual_return","sharpe","max_drawdown"]},ensure_ascii=False),flush=True)
    finally:con.close()
    (REPORT/"result.json").write_text(json.dumps({"status":"research_only","results":rows},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
