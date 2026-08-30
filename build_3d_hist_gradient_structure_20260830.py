"""One frozen strict-PIT 3D HistGradient research-only candidate build."""
from __future__ import annotations
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from build_expanding_pit_oof_baselines_20260829 import CALENDAR_2021_2024, FEATURE_DB, FEATURE_TABLE, HORIZONS, LABEL_DB, LABEL_TABLE, POLICY_PATH, canonical_frame_hash, daily_weights, evaluate, query_features, query_labels, quote, sha256_file
from build_10d_total_mv_feature_expansion_candidate_20260830 import compare_metrics, dump_json
from incremental_formal_l4_duckdb_mainline import resolve_savedmodel_feature_aliases

CANDIDATE_ID="v261_3d_hist_gradient_regressor_v1"; CONTRACT=Path("quant/data_file/reports/model_agent_3d_hist_gradient_structure_20260830/training_contract.json"); BASELINE_ROOT=Path("quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3"); OUT=Path("quant/data_file/reports/model_agent_3d_hist_gradient_structure_20260830/build_r1"); END="20241231"
def sf(frame,col): return frame.rename(columns={col:"pred_prob"})[["trade_date","stock_code","target","pred_prob"]]
def run():
 if OUT.exists(): raise RuntimeError("blocked_existing_output")
 OUT.mkdir(parents=True); contract=json.loads(CONTRACT.read_text(encoding="utf-8")); assert contract["candidate_id"]==CANDIDATE_ID
 policy=json.loads(POLICY_PATH.read_text(encoding="utf-8"));
 if policy["training_window_hard_constraints"]["mode"]!="expanding_available_history": raise RuntimeError("blocked_training_window_policy")
 c=duckdb.connect(str(CALENDAR_2021_2024),read_only=True); dates=[str(x[0]) for x in c.execute("SELECT cal_date FROM official_trade_cal WHERE is_open=1 AND cal_date<=? ORDER BY cal_date",[END]).fetchall()]; c.close()
 if not dates or dates[-1]!=END or dates!=sorted(set(dates)): raise RuntimeError("blocked_calendar")
 h=next(x for x in HORIZONS if x.key=="3d"); meta=json.loads(h.metadata_path.read_text(encoding="utf-8")); fc=duckdb.connect(str(FEATURE_DB),read_only=True); lc=duckdb.connect(str(LABEL_DB),read_only=True)
 try:
  fmin=str(fc.execute(f"SELECT min(trade_date) FROM {quote(FEATURE_TABLE)} WHERE trade_date<=?",[END]).fetchone()[0]); lmin=str(lc.execute(f"SELECT min(trade_date) FROM {quote(LABEL_TABLE)} WHERE trade_date<=?",[END]).fetchone()[0]); start=max(fmin,lmin); available={str(r[1]) for r in fc.execute(f"PRAGMA table_info({quote(FEATURE_TABLE)})").fetchall()}; requested=[str(x) for x in meta["feature_columns"]]; aliases=resolve_savedmodel_feature_aliases(requested,available)
  if aliases["missing"]: raise RuntimeError(f"blocked_aliases:{aliases['missing']}")
  amap=dict(aliases["alias_pairs"]); feats=[amap.get(x,x) for x in requested]
  if len(feats)!=15 or len(set(feats))!=15: raise RuntimeError("blocked_feature_contract")
  folds=[]
  for year in ("2022","2023","2024"):
   y=[d for d in dates if d.startswith(year)]; i=dates.index(y[0]); folds.append({"fold_id":f"fold{year}","train_start":start,"train_end":dates[i-h.maturity_sessions],"test_start":y[0],"test_end":dates[dates.index(y[-1])-h.maturity_sessions],"embargo_rule":f"{h.maturity_sessions} official open sessions"})
  dump_json(OUT/"preflight.json",{"candidate_id":CANDIDATE_ID,"runtime_agent_route":{"model":"gpt-5.6-terra","thinking":"medium"},"development_window":["20220101","20241231"],"sealed_windows":{"2025":"not_read","2026_plus":"not_read"},"feature_db_sha256":sha256_file(FEATURE_DB),"label_db_sha256":sha256_file(LABEL_DB),"calendar_sha256":sha256_file(CALENDAR_2021_2024),"production_metadata_sha256":sha256_file(h.metadata_path),"features":feats,"feature_alias_pairs":aliases["alias_pairs"],"folds":folds,"production_unchanged":True})
  rows=[]; results={}
  for fold in folds:
   tx=query_features(fc,feats,fold["train_start"],fold["train_end"]);ty=query_labels(lc,h.label,fold["train_start"],fold["train_end"]);train=tx.merge(ty,on=["trade_date","stock_code"],how="inner",validate="one_to_one").dropna(subset=["target"]);w=daily_weights(train[["trade_date","stock_code","target"]],meta.get("sample_weight_config"));vx=query_features(fc,feats,fold["test_start"],fold["test_end"]);vy=query_labels(lc,h.label,fold["test_start"],fold["test_end"]);test=vx.merge(vy,on=["trade_date","stock_code"],how="inner",validate="one_to_one").dropna(subset=["target"])
   model=HistGradientBoostingRegressor(loss="squared_error",learning_rate=.05,max_iter=300,max_leaf_nodes=31,l2_regularization=8.,early_stopping=False,random_state=42);a=train[feats].astype("float32");b=test[feats].astype("float32");model.fit(a,train.target.astype("float32"),sample_weight=w);preds=[model.predict(b).astype("float64") for _ in range(3)]
   if not(np.array_equal(preds[0],preds[1]) and np.array_equal(preds[0],preds[2])): raise RuntimeError(f"blocked_deterministic:{fold['fold_id']}")
   e=test[["trade_date","stock_code","target"]].copy();e["candidate_pred_prob"]=preds[0]; con=duckdb.connect(); base=con.execute(f"SELECT trade_date,stock_code,pred_prob AS baseline_pred_prob FROM read_parquet('{BASELINE_ROOT.as_posix()}/3d_oof.parquet') WHERE fold_id=? AND label_mature_within_dev AND trade_date BETWEEN ? AND ?",[fold["fold_id"],fold["test_start"],fold["test_end"]]).fetchdf();con.close();e=e.merge(base,on=["trade_date","stock_code"],how="inner",validate="one_to_one")
   if len(e)!=len(test) or e.duplicated(["trade_date","stock_code"]).any() or e.stock_code.astype(str).str.endswith(".BJ").any() or not np.isfinite(e[["target","baseline_pred_prob","candidate_pred_prob"]].to_numpy(dtype="float64")).all(): raise RuntimeError(f"blocked_quality:{fold['fold_id']}")
   bm,_=evaluate(sf(e,"baseline_pred_prob"));cm,_=evaluate(sf(e,"candidate_pred_prob"));results[fold["fold_id"]]={"fold":fold,"train_rows":int(len(train)),"test_rows":int(len(e)),"baseline_metrics":bm,"candidate_metrics":cm,"failed_gates":compare_metrics(bm,cm),"candidate_prediction_sha256":canonical_frame_hash(sf(e,"candidate_pred_prob"),["trade_date","stock_code","pred_prob"])};e["fold_id"]=fold["fold_id"];rows.append(e)
 finally: fc.close();lc.close()
 oof=pd.concat(rows,ignore_index=True);bm,_=evaluate(sf(oof,"baseline_pred_prob"));cm,_=evaluate(sf(oof,"candidate_pred_prob"));fails=sorted(set(compare_metrics(bm,cm)+[z for r in results.values() for z in r["failed_gates"]]));passed=not fails;oof.to_parquet(OUT/"baseline_candidate_same_key_oof.parquet",index=False)
 summary={"candidate_id":CANDIDATE_ID,"status":"completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search","development_window":["20220101","20241231"],"sealed_windows":{"2025":"not_read","2026_plus":"not_read"},"same_key_rows":int(len(oof)),"same_key_duplicate_groups":int(oof.duplicated(["trade_date","stock_code"]).sum()),"bj_rows":int(oof.stock_code.astype(str).str.endswith(".BJ").sum()),"null_or_nonfinite_rows":int((~np.isfinite(oof[["target","baseline_pred_prob","candidate_pred_prob"]].to_numpy(dtype="float64")).all(axis=1)).sum()),"aggregate":{"baseline_metrics":bm,"candidate_metrics":cm,"failed_gates":compare_metrics(bm,cm)},"folds":results,"hard_gate_passed":passed,"failed_gates":fails,"baseline_score_sha256":canonical_frame_hash(sf(oof,"baseline_pred_prob"),["trade_date","stock_code","pred_prob"]),"candidate_score_sha256":canonical_frame_hash(sf(oof,"candidate_pred_prob"),["trade_date","stock_code","pred_prob"]),"production_unchanged":True,"allow_next_layer_continue":False};dump_json(OUT/"evaluation_summary.json",summary);dump_json(OUT/"research_candidate_manifest.json",{"candidate_id":CANDIDATE_ID,"approval_status":"research_only_not_for_l5","decision":summary["status"],"production_unchanged":True,"allow_next_layer_continue":False});dump_json(OUT/"hash_inventory.json",{"script_sha256":sha256_file(Path(__file__)),"contract_sha256":sha256_file(CONTRACT),"preflight_sha256":sha256_file(OUT/"preflight.json"),"oof_sha256":sha256_file(OUT/"baseline_candidate_same_key_oof.parquet"),"summary_sha256":sha256_file(OUT/"evaluation_summary.json")});return 0
if __name__=="__main__":raise SystemExit(run())
