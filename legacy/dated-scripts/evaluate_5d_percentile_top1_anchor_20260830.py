"""Frozen 5D top1-anchored, Top10-internal percentile OOF ranking candidate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_expanding_pit_oof_baselines_20260829 import canonical_frame_hash,evaluate,sha256_file
from build_10d_total_mv_feature_expansion_candidate_20260830 import compare_metrics,dump_json
CANDIDATE_ID="v262_5d_percentile_top1_anchor_top10_internal_rank_v1";CONTRACT=Path("quant/data_file/reports/model_agent_5d_percentile_top1_anchor_top10_internal_rank_20260830/training_contract.json");BASE=Path("quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3/5d_oof.parquet");SRC=Path("quant/data_file/reports/model_agent_5d_daily_percentile_target_dev2022_2024_20260830/build_r1/candidate_oof.parquet");OUT=Path("quant/data_file/reports/model_agent_5d_percentile_top1_anchor_top10_internal_rank_20260830/build_r1")
def sf(x,c):return x.rename(columns={c:"pred_prob"})[["trade_date","stock_code","target","pred_prob"]]
def day(g):
 b=g.sort_values(["baseline_pred_prob","stock_code"],ascending=[False,True],kind="mergesort").copy(); anchor=b.head(1); movable=b.iloc[1:10].sort_values(["percentile_pred_prob","stock_code"],ascending=[False,True],kind="mergesort"); result=pd.concat([anchor,movable,b.iloc[10:]],ignore_index=True)
 if result.iloc[0].stock_code!=b.iloc[0].stock_code or set(result.head(10).stock_code)!=set(b.head(10).stock_code):raise RuntimeError("blocked_top1_or_top10_membership")
 result["baseline_rank"]=result.stock_code.map({code:i+1 for i,code in enumerate(b.stock_code)}).astype("int32");result["final_rank"]=np.arange(1,len(result)+1,dtype="int32");result["candidate_raw_score"]=(len(result)-result.final_rank).astype("float64");return result
def project(x):return pd.concat([day(g) for _,g in x.groupby("trade_date",sort=True)],ignore_index=True).sort_values(["trade_date","stock_code"],kind="mergesort").reset_index(drop=True)
def run():
 if OUT.exists():raise RuntimeError("blocked_existing_output")
 OUT.mkdir(parents=True);c=json.loads(CONTRACT.read_text(encoding="utf-8"));
 if c["candidate_id"]!=CANDIDATE_ID:raise RuntimeError("blocked_contract")
 p=pd.read_parquet(SRC);b=pd.read_parquet(BASE);b=b[(b.label_mature_within_dev)&b.fold_id.isin(["fold2022","fold2023","fold2024"])][["trade_date","stock_code","target","fold_id","pred_prob"]].rename(columns={"pred_prob":"baseline_pred_prob"});s=p[["trade_date","stock_code","target","fold_id","pred_prob"]].rename(columns={"pred_prob":"percentile_pred_prob"});x=s.merge(b,on=["trade_date","stock_code","fold_id"],how="inner",suffixes=("","_base"),validate="one_to_one")
 if len(x)!=len(s) or not np.allclose(x.target,x.target_base,equal_nan=False):raise RuntimeError("blocked_same_key_or_target")
 x=x.drop(columns="target_base")
 if x.trade_date.astype(str).str[:4].isin(["2025","2026"]).any() or x.duplicated(["trade_date","stock_code"]).any() or x.stock_code.astype(str).str.endswith(".BJ").any() or not np.isfinite(x[["target","baseline_pred_prob","percentile_pred_prob"]].to_numpy(dtype="float64")).all():raise RuntimeError("blocked_source_quality")
 dump_json(OUT/"preflight.json",{"candidate_id":CANDIDATE_ID,"contract_sha256":sha256_file(CONTRACT),"baseline_oof_sha256":sha256_file(BASE),"percentile_oof_sha256":sha256_file(SRC),"development_window":["20220101","20241231"],"sealed_windows":{"2025":"not_read","2026_plus":"not_read"},"production_unchanged":True})
 reps=[project(x) for _ in range(3)];hs=[canonical_frame_hash(z,["trade_date","stock_code","candidate_raw_score"]) for z in reps]
 if len(set(hs))!=1:raise RuntimeError("blocked_deterministic")
 oof=reps[0];bm,_=evaluate(sf(oof,"baseline_pred_prob"));cm,_=evaluate(sf(oof,"candidate_raw_score"));folds={}
 for fid,g in oof.groupby("fold_id",sort=True):
  bb,_=evaluate(sf(g,"baseline_pred_prob"));cc,_=evaluate(sf(g,"candidate_raw_score"));folds[str(fid)]={"rows":int(len(g)),"baseline_metrics":bb,"candidate_metrics":cc,"failed_gates":compare_metrics(bb,cc)}
 fails=compare_metrics(bm,cm)+[z for r in folds.values() for z in r["failed_gates"]];changed=float((oof.baseline_rank!=oof.final_rank).mean());
 if changed==0:fails.append("nonzero_effective_change")
 fails=sorted(set(fails));passed=not fails;oof.to_parquet(OUT/"baseline_candidate_same_key_oof.parquet",index=False)
 summary={"candidate_id":CANDIDATE_ID,"status":"completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search","development_window":["20220101","20241231"],"sealed_windows":{"2025":"not_read","2026_plus":"not_read"},"same_key_rows":int(len(oof)),"same_key_duplicate_groups":int(oof.duplicated(["trade_date","stock_code"]).sum()),"bj_rows":int(oof.stock_code.astype(str).str.endswith(".BJ").sum()),"null_or_nonfinite_rows":int((~np.isfinite(oof[["target","baseline_pred_prob","candidate_raw_score"]].to_numpy(dtype="float64")).all(axis=1)).sum()),"projection":{"production_top1_anchored":True,"top10_membership_identical":True,"changed_rank_row_share":changed},"aggregate":{"baseline_metrics":bm,"candidate_metrics":cm,"failed_gates":compare_metrics(bm,cm)},"folds":folds,"hard_gate_passed":passed,"failed_gates":fails,"deterministic":{"passed":True,"replay_candidate_score_sha256":hs},"baseline_score_sha256":canonical_frame_hash(sf(oof,"baseline_pred_prob"),["trade_date","stock_code","pred_prob"]),"candidate_score_sha256":canonical_frame_hash(sf(oof,"candidate_raw_score"),["trade_date","stock_code","pred_prob"]),"production_unchanged":True,"allow_next_layer_continue":False};dump_json(OUT/"evaluation_summary.json",summary);dump_json(OUT/"research_candidate_manifest.json",{"candidate_id":CANDIDATE_ID,"approval_status":"research_only_not_for_l5","decision":summary["status"],"production_unchanged":True,"allow_next_layer_continue":False});dump_json(OUT/"hash_inventory.json",{"script_sha256":sha256_file(Path(__file__)),"contract_sha256":sha256_file(CONTRACT),"baseline_oof_sha256":sha256_file(BASE),"source_percentile_oof_sha256":sha256_file(SRC),"oof_sha256":sha256_file(OUT/"baseline_candidate_same_key_oof.parquet"),"summary_sha256":sha256_file(OUT/"evaluation_summary.json")});return 0
if __name__=="__main__":raise SystemExit(run())
