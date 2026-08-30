"""Frozen research-only 1D daily top-5-percent classifier candidate."""
from __future__ import annotations
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd, xgboost as xgb
import build_expanding_pit_oof_baselines_20260829 as base

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'quant'/'data_file'
OUT=DATA/'reports'/'model_agent_1d_daily_top5_classifier_20260829'/'build_r1'
CONTRACT=OUT.parent/'training_contract.json'; BASE=DATA/'reports'/'model_agent_expanding_pit_oof_baselines_20260829_r3'

def top5(frame: pd.DataFrame)->pd.Series:
    ranks=frame.groupby('trade_date',sort=False)['target'].rank(method='first',ascending=False)
    counts=frame.groupby('trade_date',sort=False)['target'].transform('count')
    return (ranks<=np.maximum(1,np.floor(counts*0.05))).astype('int8')

def metrics(frame):
    result,_=base.evaluate(frame)
    return {k:float(result[k]) for k in ('rank_ic_mean','top1_excess_mean','top3_excess_mean','top5_excess_mean','top10_excess_mean')}

def main()->int:
    if OUT.exists(): raise RuntimeError(f'refusing to overwrite {OUT}')
    c=json.loads(CONTRACT.read_text(encoding='utf-8'))
    if c['candidate_id']!='expanding_pit_1d_daily_top5_classifier_v1' or not c['validation_2026_closed']: raise RuntimeError('invalid frozen contract')
    dates=base.read_open_dates(); _, prepared=base.preflight(dates); info=prepared['1d']; meta=info['metadata']
    baseline=pd.read_parquet(BASE/'1d_oof.parquet')
    if baseline.trade_date.max()>base.DEVELOPMENT_END: raise RuntimeError('boundary violation')
    OUT.mkdir(parents=True)
    base.dump_json(OUT/'preflight.json',{'contract_sha256':base.sha256_file(CONTRACT),'baseline_sha256':base.sha256_file(BASE/'1d_oof.parquet'),'production_metadata_sha256':base.sha256_file(base.HORIZONS[0].metadata_path),'folds':info['folds'],'development_closed_after':base.DEVELOPMENT_END,'validation_2026_closed':True,'runtime_agent_route':{'model':'gpt-5.6-terra','thinking':'medium'},'production_unchanged':True})
    fc=duckdb.connect(str(base.FEATURE_DB),read_only=True); lc=duckdb.connect(str(base.LABEL_DB),read_only=True); frames=[]
    try:
      for fold in info['folds']:
        train=base.query_features(fc,info['resolved'],fold['train_start'],fold['train_end']).merge(base.query_labels(lc,'executable_1d_open_return',fold['train_start'],fold['train_end']),on=['trade_date','stock_code'],how='inner',validate='one_to_one').dropna(subset=['target']).copy()
        test=base.query_features(fc,info['resolved'],fold['test_start'],fold['test_end']).merge(base.query_labels(lc,'executable_1d_open_return',fold['test_start'],fold['mature_label_cutoff']),on=['trade_date','stock_code'],how='left',validate='one_to_one')
        p=base.model_params(meta); p.update({'objective':'binary:logistic','eval_metric':'logloss'})
        model=xgb.XGBClassifier(**p); y=top5(train); model.fit(train[info['resolved']].astype('float32'),y,verbose=False)
        a=model.predict_proba(test[info['resolved']].astype('float32'))[:,1].astype('float64'); b=model.predict_proba(test[info['resolved']].astype('float32'))[:,1].astype('float64'); d=model.predict_proba(test[info['resolved']].astype('float32'))[:,1].astype('float64')
        if not(np.array_equal(a,b) and np.array_equal(a,d)): raise RuntimeError(f"{fold['fold_id']}: deterministic replay failed")
        f=test[['trade_date','stock_code','target']].copy(); f['pred_prob']=a; f['fold_id']=fold['fold_id']; f['label_mature_within_dev']=f.trade_date<=fold['mature_label_cutoff']
        if f.duplicated(['trade_date','stock_code']).any() or f.stock_code.str.endswith('.BJ').any() or f.pred_prob.isna().any() or not np.isfinite(f.pred_prob).all(): raise RuntimeError(f"{fold['fold_id']}: quality gate failed")
        frames.append(f)
    finally: fc.close(); lc.close()
    cand=pd.concat(frames,ignore_index=True); b=baseline.loc[baseline.label_mature_within_dev,['trade_date','stock_code','target','pred_prob','fold_id']].reset_index(drop=True); x=cand.loc[cand.label_mature_within_dev,['trade_date','stock_code','target','pred_prob','fold_id']].reset_index(drop=True)
    if not b[['trade_date','stock_code']].equals(x[['trade_date','stock_code']]): raise RuntimeError('same-key closure failed')
    folds=[]
    for fid in sorted(x.fold_id.unique()):
      left,right=metrics(b.query('fold_id==@fid')),metrics(x.query('fold_id==@fid')); folds.append({'fold_id':fid,'baseline':left,'candidate':right,'delta':{k:right[k]-left[k] for k in left}})
    left,right=metrics(b),metrics(x); delta={k:right[k]-left[k] for k in left}; gate={'aggregate_rank_ic_not_weaker':delta['rank_ic_mean']>=0,'each_fold_rank_ic_not_weaker':all(z['delta']['rank_ic_mean']>=0 for z in folds),'aggregate_top1_top3_top5_top10_not_weaker':all(delta[f'top{i}_excess_mean']>=0 for i in (1,3,5,10)),'each_fold_top10_not_weaker':all(z['delta']['top10_excess_mean']>=0 for z in folds)}
    cand.to_parquet(OUT/'candidate_oof.parquet',index=False); summary={'candidate_id':c['candidate_id'],'approval_status':'research_only_not_for_l5','same_key_rows':int(len(x)),'fold_comparisons':folds,'aggregate_baseline':left,'aggregate_candidate':right,'aggregate_delta':delta,'acceptance_gate':gate,'decision':'completed_ready_for_audit' if all(gate.values()) else 'reject_no_further_search','ready_for_audit_review':True,'allow_next_layer_continue':False,'production_unchanged':True,'validation_2026_closed':True}; base.dump_json(OUT/'evaluation_summary.json',summary); base.dump_json(OUT/'audit_handoff.json',{'status':summary['decision'],'candidate_oof_sha256':base.sha256_file(OUT/'candidate_oof.parquet'),'summary_sha256':base.sha256_file(OUT/'evaluation_summary.json'),'contract_sha256':base.sha256_file(CONTRACT),'ready_for_audit_review':True,'allow_next_layer_continue':False,'production_unchanged':True})
    return 0
if __name__=='__main__': raise SystemExit(main())
