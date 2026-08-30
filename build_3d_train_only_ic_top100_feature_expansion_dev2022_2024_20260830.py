"""Strict PIT 3D train-only Top100 feature-expansion research candidate."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import duckdb, numpy as np, pandas as pd, xgboost as xgb
import build_expanding_pit_oof_baselines_20260829 as base
from fast_feature_selection import score_features_fast_split, write_score_csv

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'quant'/'data_file'; OUT=DATA/'reports'/'model_agent_3d_train_only_ic_top100_feature_expansion_dev2022_2024_20260830'/'build_r2'; CONTRACT=DATA/'reports'/'model_agent_3d_train_only_ic_top100_feature_expansion_dev2022_2024_20260830'/'training_contract.json'; BASE=DATA/'reports'/'model_agent_expanding_pit_oof_baselines_20260829_r3'
TOP_N=100; MIN_IC=.005; MAX_MISSING=.35; IC_FOLDS=8
def ms(f):
 r,_=base.evaluate(f);return {k:float(r[k]) for k in ('rank_ic_mean','top1_excess_mean','top3_excess_mean','top5_excess_mean','top10_excess_mean')}
def main():
 if OUT.exists():raise RuntimeError(f'refusing overwrite: {OUT}')
 c=json.loads(CONTRACT.read_text(encoding='utf8'));s=c['fixed_selection']
 if c['candidate_id']!='expanding_pit_3d_train_only_ic_top100_feature_expansion_v1' or (s['top_n'],s['min_abs_ic'],s['max_missing_ratio'],s['ic_folds'])!=(TOP_N,MIN_IC,MAX_MISSING,IC_FOLDS):raise RuntimeError('contract drift')
 dates=base.read_open_dates();_,pre=base.preflight(dates);info=pre['3d'];folds=[f for f in info['folds'] if f['test_end']<='20241231']
 if [f['fold_id'] for f in folds]!=['fold2022','fold2023','fold2024']:raise RuntimeError('wrong folds')
 b=pd.read_parquet(BASE/'3d_oof.parquet',filters=[('trade_date','<=','20241231')]);
 if b.trade_date.max()>'20241231':raise RuntimeError('confirmation read')
 OUT.mkdir(parents=True);sel_dir=OUT/'feature_selection';sel_dir.mkdir();base.dump_json(OUT/'preflight.json',{'contract_sha256':base.sha256_file(CONTRACT),'folds':folds,'selection':s,'read_2025_or_later':False,'production_unchanged':True,'runtime_agent_route':c['runtime_agent_route']})
 fd=duckdb.connect(str(base.FEATURE_DB),read_only=True);ld=duckdb.connect(str(base.LABEL_DB),read_only=True);rows=[];selections=[]
 try:
  for f in folds:
   scores,features=score_features_fast_split(base.FEATURE_DB,label_path=base.LABEL_DB,feature_table=base.FEATURE_TABLE,label_table=base.LABEL_TABLE,label='executable_3d_open_return',start=f['train_start'],end=f['train_end'],top_n=TOP_N,min_abs_ic=MIN_IC,max_missing_ratio=MAX_MISSING,folds=IC_FOLDS)
   if not features or len(features)>TOP_N:raise RuntimeError(f"{f['fold_id']}: invalid selected features")
   from leakage_guard import validate_no_leakage
   validate_no_leakage(features,label='executable_3d_open_return');write_score_csv(scores,sel_dir/f"{f['fold_id']}_ic_scores.csv");base.dump_json(sel_dir/f"{f['fold_id']}_features.json",{'features':features,'count':len(features),'train_start':f['train_start'],'train_end':f['train_end'],'score_rows':len(scores)})
   tr=base.query_features(fd,features,f['train_start'],f['train_end']).merge(base.query_labels(ld,'executable_3d_open_return',f['train_start'],f['train_end']),on=['trade_date','stock_code'],how='inner',validate='one_to_one').dropna(subset=['target']).reset_index(drop=True)
   te=base.query_features(fd,features,f['test_start'],f['test_end']).merge(base.query_labels(ld,'executable_3d_open_return',f['test_start'],f['mature_label_cutoff']),on=['trade_date','stock_code'],how='left',validate='one_to_one')
   m=xgb.XGBRegressor(**base.model_params(info['metadata']));m.fit(tr[features].astype('float32'),tr.target.astype('float32'),sample_weight=base.daily_weights(tr,info['metadata']['sample_weight_config']),verbose=False);a,b2,d=(m.predict(te[features].astype('float32')).astype('float64') for _ in range(3))
   if not(np.array_equal(a,b2) and np.array_equal(a,d)):raise RuntimeError(f"{f['fold_id']}: nondeterministic")
   z=te[['trade_date','stock_code','target']].copy();z['pred_prob']=a;z['fold_id']=f['fold_id'];z['label_mature_within_dev']=z.trade_date<=f['mature_label_cutoff'];
   if z.duplicated(['trade_date','stock_code']).any() or z.stock_code.astype(str).str.endswith('.BJ').any() or z.pred_prob.isna().any() or not np.isfinite(z.pred_prob).all():raise RuntimeError(f"{f['fold_id']}: quality")
   rows.append(z);selections.append({'fold_id':f['fold_id'],'feature_count':len(features),'features_sha256':hashlib.sha256(('\n'.join(features)+'\n').encode('utf-8')).hexdigest()})
 finally:fd.close();ld.close()
 q=pd.concat(rows,ignore_index=True);q.to_parquet(OUT/'candidate_oof.parquet',index=False);be=b.loc[b.label_mature_within_dev,['trade_date','stock_code','target','pred_prob','fold_id']].reset_index(drop=True);ce=q.loc[q.label_mature_within_dev,['trade_date','stock_code','target','pred_prob','fold_id']].reset_index(drop=True)
 if not be[['trade_date','stock_code']].equals(ce[['trade_date','stock_code']]):raise RuntimeError('same key')
 fs=[]
 for fid in ('fold2022','fold2023','fold2024'):
  x,y=ms(be.query('fold_id==@fid')),ms(ce.query('fold_id==@fid'));fs.append({'fold_id':fid,'baseline':x,'candidate':y,'delta':{k:y[k]-x[k] for k in x}})
 x,y=ms(be),ms(ce);delta={k:y[k]-x[k] for k in x};gate={'aggregate_rank_ic_not_weaker':delta['rank_ic_mean']>=0,'each_fold_rank_ic_not_weaker':all(r['delta']['rank_ic_mean']>=0 for r in fs),'aggregate_top1_top3_top5_top10_not_weaker':all(delta[f'top{k}_excess_mean']>=0 for k in (1,3,5,10)),'each_fold_top10_not_weaker':all(r['delta']['top10_excess_mean']>=0 for r in fs)};dec='development_pass_waiting_for_2025_confirmation' if all(gate.values()) else 'reject_no_further_search'
 summary={'candidate_id':c['candidate_id'],'stage':'development_2022_2024_only','same_key_rows':len(ce),'selections':selections,'fold_comparisons':fs,'aggregate_baseline':x,'aggregate_candidate':y,'aggregate_delta':delta,'acceptance_gate':gate,'decision':dec,'confirmation_2025_read':False,'validation_2026_closed':True,'production_unchanged':True,'allow_next_layer_continue':False};base.dump_json(OUT/'evaluation_summary.json',summary);base.dump_json(OUT/'audit_handoff.json',{'status':dec,'candidate_oof_sha256':base.sha256_file(OUT/'candidate_oof.parquet'),'summary_sha256':base.sha256_file(OUT/'evaluation_summary.json'),'contract_sha256':base.sha256_file(CONTRACT),'confirmation_2025_read':False,'allow_next_layer_continue':False,'production_unchanged':True})
if __name__=='__main__':main()
