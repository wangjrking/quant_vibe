"""One frozen research-only 3D HGB ordering inside the production Top10."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_expanding_pit_oof_baselines_20260829 import canonical_frame_hash,evaluate,sha256_file
from build_10d_total_mv_feature_expansion_candidate_20260830 import compare_metrics,dump_json
from evaluate_1d_hist_top10_internal_rank_20260830 import metric_frame,project
CANDIDATE_ID="v261_3d_hist_top10_internal_rank_v1";CONTRACT=Path("quant/data_file/reports/model_agent_3d_hist_top10_internal_rank_20260830/training_contract.json");SRC=Path("quant/data_file/reports/model_agent_3d_hist_gradient_structure_20260830/build_r1/baseline_candidate_same_key_oof.parquet");OUT=Path("quant/data_file/reports/model_agent_3d_hist_top10_internal_rank_20260830/build_r1")
def run():
 if OUT.exists():raise RuntimeError("blocked_existing_output")
 OUT.mkdir(parents=True);c=json.loads(CONTRACT.read_text(encoding="utf-8"));
 if c["candidate_id"]!=CANDIDATE_ID or c["algorithm"]["mutable_prefix_size"]!=10:raise RuntimeError("blocked_contract")
 s=pd.read_parquet(SRC);required={"trade_date","stock_code","target","baseline_pred_prob","candidate_pred_prob","fold_id"}
 if not required.issubset(s.columns) or s.trade_date.astype(str).str[:4].isin(["2025","2026"]).any() or s.duplicated(["trade_date","stock_code"]).any() or s.stock_code.astype(str).str.endswith(".BJ").any() or not np.isfinite(s[["target","baseline_pred_prob","candidate_pred_prob"]].to_numpy(dtype="float64")).all():raise RuntimeError("blocked_source")
 dump_json(OUT/"preflight.json",{"candidate_id":CANDIDATE_ID,"contract_sha256":sha256_file(CONTRACT),"hgb_oof_sha256":sha256_file(SRC),"development_window":["20220101","20241231"],"sealed_windows":{"2025":"not_read","2026_plus":"not_read"},"production_unchanged":True})
 reps=[project(s) for _ in range(3)];hashes=[canonical_frame_hash(x,["trade_date","stock_code","candidate_raw_score"]) for x in reps]
 if len(set(hashes))!=1:raise RuntimeError("blocked_deterministic")
 oof=reps[0];bm,_=evaluate(metric_frame(oof,"baseline_pred_prob"));cm,_=evaluate(metric_frame(oof,"candidate_raw_score"));folds={}
 for fid,g in oof.groupby("fold_id",sort=True):
  b,_=evaluate(metric_frame(g,"baseline_pred_prob"));d,_=evaluate(metric_frame(g,"candidate_raw_score"));folds[str(fid)]={"rows":int(len(g)),"baseline_metrics":b,"candidate_metrics":d,"failed_gates":compare_metrics(b,d)}
 fails=compare_metrics(bm,cm)+[z for r in folds.values() for z in r["failed_gates"]];changed=float((oof.baseline_rank!=oof.final_rank).mean());
 if changed==0:fails.append("nonzero_effective_change")
 fails=sorted(set(fails));passed=not fails;oof.to_parquet(OUT/"baseline_candidate_same_key_oof.parquet",index=False)
 summary={"candidate_id":CANDIDATE_ID,"status":"completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search","development_window":["20220101","20241231"],"sealed_windows":{"2025":"not_read","2026_plus":"not_read"},"same_key_rows":int(len(oof)),"same_key_duplicate_groups":int(oof.duplicated(["trade_date","stock_code"]).sum()),"bj_rows":int(oof.stock_code.astype(str).str.endswith(".BJ").sum()),"null_or_nonfinite_rows":int((~np.isfinite(oof[["target","baseline_pred_prob","candidate_raw_score"]].to_numpy(dtype="float64")).all(axis=1)).sum()),"projection":{"mutable_prefix_size":10,"changed_rank_row_share":changed,"top10_membership_identical":True},"aggregate":{"baseline_metrics":bm,"candidate_metrics":cm,"failed_gates":compare_metrics(bm,cm)},"folds":folds,"hard_gate_passed":passed,"failed_gates":fails,"deterministic":{"passed":True,"replay_candidate_score_sha256":hashes},"baseline_score_sha256":canonical_frame_hash(metric_frame(oof,"baseline_pred_prob"),["trade_date","stock_code","pred_prob"]),"candidate_score_sha256":canonical_frame_hash(metric_frame(oof,"candidate_raw_score"),["trade_date","stock_code","pred_prob"]),"production_unchanged":True,"allow_next_layer_continue":False};dump_json(OUT/"evaluation_summary.json",summary);dump_json(OUT/"research_candidate_manifest.json",{"candidate_id":CANDIDATE_ID,"approval_status":"research_only_not_for_l5","decision":summary["status"],"production_unchanged":True,"allow_next_layer_continue":False});dump_json(OUT/"hash_inventory.json",{"script_sha256":sha256_file(Path(__file__)),"contract_sha256":sha256_file(CONTRACT),"source_hgb_oof_sha256":sha256_file(SRC),"oof_sha256":sha256_file(OUT/"baseline_candidate_same_key_oof.parquet"),"summary_sha256":sha256_file(OUT/"evaluation_summary.json")});return 0
if __name__=="__main__":raise SystemExit(run())
