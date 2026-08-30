"""Strict 2022-2024-only 3D depth-3 capacity research candidate."""
from __future__ import annotations
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd, xgboost as xgb
import build_expanding_pit_oof_baselines_20260829 as base
ROOT=Path(__file__).resolve().parents[2]; DATA_DIR=ROOT/'quant'/'data_file'; OUT=DATA_DIR/'reports'/'model_agent_3d_depth3_capacity_dev2022_2024_20260830'/'build_r1'; CONTRACT=OUT.parent/'training_contract.json'; BASE=DATA_DIR/'reports'/'model_agent_expanding_pit_oof_baselines_20260829_r3'
def metrics(f):
 r,_=base.evaluate(f); return {k:float(r[k]) for k in ('rank_ic_mean','top1_excess_mean','top3_excess_mean','top5_excess_mean','top10_excess_mean')}
def main():
 if OUT.exists(): raise RuntimeError(f'refusing overwrite: {OUT}')
 c=json.loads(CONTRACT.read_text(encoding='utf-8'))
 if c['candidate_id']!='expanding_pit_3d_depth3_capacity_v1' or c['single_change']['kind']!='increase_tree_depth_2_to_3' or c['selection_and_sealing']['confirmation_window']!='2025 sealed_for_3d_candidate_only' or c['selection_and_sealing']['validation_window']!='2026 closed_for_3d_candidate_only': raise RuntimeError('contract drift')
 dates=base.read_open_dates(); _,prepared=base.preflight(dates); info=prepared['3d']; folds=[x for x in info['folds'] if x['test_end']<='20241231']
 if [x['fold_id'] for x in folds]!=['fold2022','fold2023','fold2024']: raise RuntimeError('wrong development folds')
 baseline=pd.read_parquet(BASE/'3d_oof.parquet',filters=[('trade_date','<=','20241231')])
 if baseline.trade_date.max()>'20241231': raise RuntimeError('confirmation read')
 OUT.mkdir(parents=True); base.dump_json(OUT/'preflight.json',{'contract_sha256':base.sha256_file(CONTRACT),'features':info['resolved'],'folds':folds,'production_weight_config':info['metadata']['sample_weight_config'],'override':{'max_depth':3,'n_estimators':base.model_params(info['metadata'])['n_estimators']},'confirmation_2025_read':False,'validation_2026_closed':True,'production_unchanged':True,'runtime_agent_route':c['runtime_agent_route']})
 fd=duckdb.connect(str(base.FEATURE_DB),read_only=True);ld=duckdb.connect(str(base.LABEL_DB),read_only=True); rows=[]
 try:
  for fold in folds:
   tr=base.query_features(fd,info['resolved'],fold['train_start'],fold['train_end']).merge(base.query_labels(ld,'executable_3d_open_return',fold['train_start'],fold['train_end']),on=['trade_date','stock_code'],how='inner',validate='one_to_one').dropna(subset=['target']).reset_index(drop=True)
   te=base.query_features(fd,info['resolved'],fold['test_start'],fold['test_end']).merge(base.query_labels(ld,'executable_3d_open_return',fold['test_start'],fold['mature_label_cutoff']),on=['trade_date','stock_code'],how='left',validate='one_to_one')
   p=base.model_params(info['metadata']);p['max_depth']=3;m=xgb.XGBRegressor(**p);m.fit(tr[info['resolved']].astype('float32'),tr.target.astype('float32'),sample_weight=base.daily_weights(tr,info['metadata']['sample_weight_config']),verbose=False)
   a,b,d=(m.predict(te[info['resolved']].astype('float32')).astype('float64') for _ in range(3))
   if not(np.array_equal(a,b) and np.array_equal(a,d)): raise RuntimeError(f"{fold['fold_id']}: deterministic failure")
   s=te[['trade_date','stock_code','target']].copy();s['pred_prob']=a;s['fold_id']=fold['fold_id'];s['label_mature_within_dev']=s.trade_date<=fold['mature_label_cutoff']
   if s.duplicated(['trade_date','stock_code']).any() or s.stock_code.astype(str).str.endswith('.BJ').any() or s.pred_prob.isna().any() or not np.isfinite(s.pred_prob).all(): raise RuntimeError(f"{fold['fold_id']}: quality failure")
   rows.append(s)
 finally: fd.close();ld.close()
 cand=pd.concat(rows,ignore_index=True);cand.to_parquet(OUT/'candidate_oof.parquet',index=False);be=baseline.loc[baseline.label_mature_within_dev,['trade_date','stock_code','target','pred_prob','fold_id']].reset_index(drop=True);ce=cand.loc[cand.label_mature_within_dev,['trade_date','stock_code','target','pred_prob','fold_id']].reset_index(drop=True)
 if not be[['trade_date','stock_code']].equals(ce[['trade_date','stock_code']]): raise RuntimeError('same key failure')
 comps=[]
 for fid in ('fold2022','fold2023','fold2024'):
  before,after=metrics(be.query('fold_id==@fid')),metrics(ce.query('fold_id==@fid'));comps.append({'fold_id':fid,'baseline':before,'candidate':after,'delta':{k:after[k]-before[k] for k in before}})
 before,after=metrics(be),metrics(ce);delta={k:after[k]-before[k] for k in before};gate={'aggregate_rank_ic_not_weaker':delta['rank_ic_mean']>=0,'each_fold_rank_ic_not_weaker':all(x['delta']['rank_ic_mean']>=0 for x in comps),'aggregate_top1_top3_top5_top10_not_weaker':all(delta[f'top{k}_excess_mean']>=0 for k in (1,3,5,10)),'each_fold_top10_not_weaker':all(x['delta']['top10_excess_mean']>=0 for x in comps)}
 summary={'candidate_id':c['candidate_id'],'stage':'development_2022_2024_only','same_key_rows':len(ce),'fold_comparisons':comps,'aggregate_baseline':before,'aggregate_candidate':after,'aggregate_delta':delta,'acceptance_gate':gate,'decision':'development_pass_waiting_for_2025_confirmation' if all(gate.values()) else 'reject_no_further_search','confirmation_2025_read':False,'validation_2026_closed':True,'production_unchanged':True,'allow_next_layer_continue':False};base.dump_json(OUT/'evaluation_summary.json',summary);base.dump_json(OUT/'audit_handoff.json',{'status':summary['decision'],'candidate_oof_sha256':base.sha256_file(OUT/'candidate_oof.parquet'),'summary_sha256':base.sha256_file(OUT/'evaluation_summary.json'),'contract_sha256':base.sha256_file(CONTRACT),'confirmation_2025_read':False,'allow_next_layer_continue':False,'production_unchanged':True})
if __name__=='__main__': main()
