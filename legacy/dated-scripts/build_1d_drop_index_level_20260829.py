"""Frozen research-only 1D candidate removing four non-cross-sectional index levels."""
from __future__ import annotations
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd, xgboost as xgb
import build_expanding_pit_oof_baselines_20260829 as base
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'quant'/'data_file'; OUT=DATA/'reports'/'model_agent_1d_drop_index_level_20260829'/'build_r1'; CONTRACT=OUT.parent/'training_contract.json'; BASE=DATA/'reports'/'model_agent_expanding_pit_oof_baselines_20260829_r3'; DROP={'index_2000_open','index_2000_high','index_2000_low','index_2000_close'}
def m(frame):
 r,_=base.evaluate(frame); return {k:float(r[k]) for k in ('rank_ic_mean','top1_excess_mean','top3_excess_mean','top5_excess_mean','top10_excess_mean')}
def main():
 if OUT.exists(): raise RuntimeError(f'refusing to overwrite {OUT}')
 c=json.loads(CONTRACT.read_text(encoding='utf-8')); dates=base.read_open_dates(); _, p=base.preflight(dates); info=p['1d']; features=[x for x in info['resolved'] if x not in DROP]
 if len(features)!=74 or not DROP.isdisjoint(set(features)): raise RuntimeError('frozen feature removal closure failed')
 baseline=pd.read_parquet(BASE/'1d_oof.parquet'); OUT.mkdir(parents=True); base.dump_json(OUT/'preflight.json',{'contract_sha256':base.sha256_file(CONTRACT),'baseline_sha256':base.sha256_file(BASE/'1d_oof.parquet'),'production_metadata_sha256':base.sha256_file(base.HORIZONS[0].metadata_path),'removed_features':sorted(DROP),'remaining_features':features,'folds':info['folds'],'development_closed_after':base.DEVELOPMENT_END,'validation_2026_closed':True,'runtime_agent_route':{'model':'gpt-5.6-terra','thinking':'medium'},'production_unchanged':True})
 fc=duckdb.connect(str(base.FEATURE_DB),read_only=True); lc=duckdb.connect(str(base.LABEL_DB),read_only=True); frames=[]
 try:
  for f in info['folds']:
   tr=base.query_features(fc,features,f['train_start'],f['train_end']).merge(base.query_labels(lc,'executable_1d_open_return',f['train_start'],f['train_end']),on=['trade_date','stock_code'],how='inner',validate='one_to_one').dropna(subset=['target']); te=base.query_features(fc,features,f['test_start'],f['test_end']).merge(base.query_labels(lc,'executable_1d_open_return',f['test_start'],f['mature_label_cutoff']),on=['trade_date','stock_code'],how='left',validate='one_to_one')
   model=xgb.XGBRegressor(**base.model_params(info['metadata'])); model.fit(tr[features].astype('float32'),tr.target.astype('float32'),verbose=False); a=model.predict(te[features].astype('float32')).astype('float64'); b=model.predict(te[features].astype('float32')).astype('float64'); d=model.predict(te[features].astype('float32')).astype('float64')
   if not(np.array_equal(a,b) and np.array_equal(a,d)): raise RuntimeError(f"{f['fold_id']}: deterministic replay failed")
   z=te[['trade_date','stock_code','target']].copy(); z['pred_prob']=a; z['fold_id']=f['fold_id']; z['label_mature_within_dev']=z.trade_date<=f['mature_label_cutoff'];
   if z.duplicated(['trade_date','stock_code']).any() or z.stock_code.str.endswith('.BJ').any() or z.pred_prob.isna().any() or not np.isfinite(z.pred_prob).all(): raise RuntimeError(f"{f['fold_id']}: quality gate failed")
   frames.append(z)
 finally: fc.close(); lc.close()
 cand=pd.concat(frames,ignore_index=True); b=baseline.loc[baseline.label_mature_within_dev,['trade_date','stock_code','target','pred_prob','fold_id']].reset_index(drop=True); x=cand.loc[cand.label_mature_within_dev,['trade_date','stock_code','target','pred_prob','fold_id']].reset_index(drop=True)
 if not b[['trade_date','stock_code']].equals(x[['trade_date','stock_code']]): raise RuntimeError('same-key closure failed')
 folds=[]
 for fid in sorted(x.fold_id.unique()):
  l,r=m(b.query('fold_id==@fid')),m(x.query('fold_id==@fid')); folds.append({'fold_id':fid,'baseline':l,'candidate':r,'delta':{k:r[k]-l[k] for k in l}})
 l,r=m(b),m(x); delta={k:r[k]-l[k] for k in l}; gate={'aggregate_rank_ic_not_weaker':delta['rank_ic_mean']>=0,'each_fold_rank_ic_not_weaker':all(z['delta']['rank_ic_mean']>=0 for z in folds),'aggregate_top1_top3_top5_top10_not_weaker':all(delta[f'top{i}_excess_mean']>=0 for i in (1,3,5,10)),'each_fold_top10_not_weaker':all(z['delta']['top10_excess_mean']>=0 for z in folds)}
 cand.to_parquet(OUT/'candidate_oof.parquet',index=False); s={'candidate_id':c['candidate_id'],'approval_status':'research_only_not_for_l5','same_key_rows':len(x),'fold_comparisons':folds,'aggregate_baseline':l,'aggregate_candidate':r,'aggregate_delta':delta,'acceptance_gate':gate,'decision':'completed_ready_for_audit' if all(gate.values()) else 'reject_no_further_search','ready_for_audit_review':True,'allow_next_layer_continue':False,'production_unchanged':True,'validation_2026_closed':True}; base.dump_json(OUT/'evaluation_summary.json',s); base.dump_json(OUT/'audit_handoff.json',{'status':s['decision'],'candidate_oof_sha256':base.sha256_file(OUT/'candidate_oof.parquet'),'summary_sha256':base.sha256_file(OUT/'evaluation_summary.json'),'contract_sha256':base.sha256_file(CONTRACT),'ready_for_audit_review':True,'allow_next_layer_continue':False,'production_unchanged':True})
if __name__=='__main__': main()
