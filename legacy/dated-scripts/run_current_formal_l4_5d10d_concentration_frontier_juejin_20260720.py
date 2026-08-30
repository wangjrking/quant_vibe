from __future__ import annotations
import importlib.util,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/"quant"/"main"/"run_current_formal_l4_5d10d_position_count_juejin_20260720.py";S=importlib.util.spec_from_file_location("pc",SRC);M=importlib.util.module_from_spec(S);assert S and S.loader;S.loader.exec_module(M)
OUT=ROOT/"quant"/"data_file"/"reports"/"strategy_agent_current_formal_l4_5d10d_concentration_frontier_juejin_20260720";M.OUT=OUT;M.SIG=OUT/"signals";M.LOG=OUT/"logs";CASES=[(r1,mp) for r1 in [.80,.85,.90] for mp in [1,2,3,4,5,7]]
def main():
 [p.mkdir(parents=True,exist_ok=True) for p in [OUT,M.SIG,M.LOG]];rows=[]
 for c in CASES:
  x=M.run(*c);rows.append(x);pd.DataFrame(rows).sort_values(["annual_return","sharpe"],ascending=False,na_position="last").to_csv(OUT/"juejin_results.csv",index=False,encoding="utf-8-sig");print(json.dumps({k:x.get(k) for k in ["name","returncode","annual_return","sharpe","max_drawdown"]},ensure_ascii=False),flush=True)
 (OUT/"result.json").write_text(json.dumps({"status":"research_only","results":rows},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
