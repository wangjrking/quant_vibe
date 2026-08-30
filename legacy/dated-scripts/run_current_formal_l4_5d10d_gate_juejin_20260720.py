from __future__ import annotations
import importlib.util,json
from pathlib import Path
import duckdb,pandas as pd
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/"quant"/"main"/"run_current_formal_l4_persistent_edge_juejin_20260720.py";S=importlib.util.spec_from_file_location("r",SRC);M=importlib.util.module_from_spec(S);assert S and S.loader;S.loader.exec_module(M)
OUT=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_5d10d_gate_juejin_20260720";M.REPORT_DIR=OUT;M.SIGNAL_DIR=OUT/"signals";M.SCORE_DIR=OUT/"score_assets";M.LOG_DIR=OUT/"logs"
for floor in [.4,.5,.6,.7,.8]:M.FILTERS[f"f{int(floor*100)}_pct3"]=f"least(rank_5d,rank_10d)>={floor} AND signal_pct_chg_raw<=3.0"
CASES=[{"blend":"w10_100","score":"plain","filter":f"f{floor}_pct3","threshold":.98,"topn":1,"hold":12} for floor in [40,50,60,70,80]]+[{"blend":"w10_100","score":"plain","filter":"f50_pct3","threshold":.98,"topn":1,"hold":15}]
def main():
 [p.mkdir(parents=True,exist_ok=True) for p in [OUT,M.SIGNAL_DIR,M.SCORE_DIR,M.LOG_DIR]];c=duckdb.connect(str(M.FEATURE_DB),read_only=True);rows=[]
 try:
  score=M.build_score(c,"w10_100")
  for case in CASES:
   n,s,mp,tp=M.build_signal(c,case);x=M.run_case(case,n,s,mp,tp,score);rows.append(x);pd.DataFrame(rows).sort_values(["sharpe","annual_return"],ascending=False,na_position="last").to_csv(OUT/"juejin_results.csv",index=False,encoding="utf-8-sig");print(json.dumps({k:x.get(k) for k in ["name","returncode","annual_return","sharpe","max_drawdown"]},ensure_ascii=False),flush=True)
 finally:c.close()
 (OUT/"result.json").write_text(json.dumps({"status":"research_only","results":rows},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
