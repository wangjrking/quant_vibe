"""Strict 2022-2024-only stationary-market-price 1D research candidate."""
from __future__ import annotations
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd, xgboost as xgb
import build_expanding_pit_oof_baselines_20260829 as base

ROOT=Path(__file__).resolve().parents[2]; DATA_DIR=ROOT/'quant'/'data_file'
OUT=DATA_DIR/'reports'/'model_agent_1d_stationary_market_price_dev2022_2024_20260830'/'build_r1'
CONTRACT=OUT.parent/'training_contract.json'; BASE=DATA_DIR/'reports'/'model_agent_expanding_pit_oof_baselines_20260829_r3'
INDEX=['index_2000_close','index_2000_low','index_2000_high','index_2000_open']

def metrics(frame):
    result,_=base.evaluate(frame)
    return {k:float(result[k]) for k in ('rank_ic_mean','top1_excess_mean','top3_excess_mean','top5_excess_mean','top10_excess_mean')}

def prior_date(conn,date):
    r=conn.execute(f"select max(trade_date) from {base.quote(base.FEATURE_TABLE)} where trade_date < ?",[date]).fetchone()[0]
    return None if r is None else str(r)

def feature_columns(info):
    return [f'{x}_to_prev_index_close' if x in INDEX else x for x in info['resolved']]

def transformed(conn, info, start,end):
    p=prior_date(conn,start); raw=base.query_features(conn,info['resolved'],p or start,end)
    for col in INDEX:
        if raw.groupby('trade_date',sort=True)[col].nunique(dropna=True).gt(1).any(): raise RuntimeError(f'{col}: multiple non-null market values within a trade date')
    daily=raw.groupby('trade_date',sort=True)[INDEX].first().reset_index(); daily['prior_index_close']=daily['index_2000_close'].shift(1)
    raw=raw.merge(daily[['trade_date','prior_index_close']],on='trade_date',how='left',validate='many_to_one')
    for col in INDEX: raw[f'{col}_to_prev_index_close']=raw[col]/raw.prior_index_close-1.0
    cols=feature_columns(info); result=raw.loc[raw.trade_date>=start,['trade_date','stock_code',*cols]].reset_index(drop=True)
    if np.isinf(result[cols].to_numpy(dtype='float64',copy=False)).any(): raise RuntimeError('relative-price transform produced infinity')
    return result

def main():
    if OUT.exists(): raise RuntimeError(f'refusing overwrite: {OUT}')
    c=json.loads(CONTRACT.read_text(encoding='utf-8'))
    if c['candidate_id']!='expanding_pit_1d_stationary_market_price_v1' or c['confirmation_window']!='2025 sealed_for_1d_candidate_only' or not c['validation_2026_closed_for_1d_candidate_only']: raise RuntimeError('contract drift')
    dates=base.read_open_dates(); _,prepared=base.preflight(dates); info=prepared['1d']; folds=[f for f in info['folds'] if f['test_end']<='20241231']; cols=feature_columns(info)
    if [f['fold_id'] for f in folds]!=['fold2022','fold2023','fold2024'] or info['metadata'].get('sample_weight_config') is not None: raise RuntimeError('wrong production contract')
    baseline=pd.read_parquet(BASE/'1d_oof.parquet',filters=[('trade_date','<=','20241231')])
    if baseline.trade_date.max()>'20241231': raise RuntimeError('confirmation read')
    OUT.mkdir(parents=True); base.dump_json(OUT/'preflight.json',{'contract_sha256':base.sha256_file(CONTRACT),'production_feature_columns':info['resolved'],'candidate_feature_columns':cols,'folds':folds,'price_transform':'index OHLC / prior available index close - 1','confirmation_2025_read':False,'validation_2026_closed':True,'production_unchanged':True,'runtime_agent_route':{'model':'gpt-5.6-terra','thinking':'medium'}})
    fd=duckdb.connect(str(base.FEATURE_DB),read_only=True); ld=duckdb.connect(str(base.LABEL_DB),read_only=True); rows=[]
    try:
        for f in folds:
            tr=transformed(fd,info,f['train_start'],f['train_end']).merge(base.query_labels(ld,'executable_1d_open_return',f['train_start'],f['train_end']),on=['trade_date','stock_code'],how='inner',validate='one_to_one').dropna(subset=['target']).reset_index(drop=True)
            te=transformed(fd,info,f['test_start'],f['test_end']).merge(base.query_labels(ld,'executable_1d_open_return',f['test_start'],f['mature_label_cutoff']),on=['trade_date','stock_code'],how='left',validate='one_to_one')
            model=xgb.XGBRegressor(**base.model_params(info['metadata'])); xtr,xte=tr[cols].astype('float32'),te[cols].astype('float32'); model.fit(xtr,tr.target.astype('float32'),verbose=False)
            a,b,d=(model.predict(xte).astype('float64') for _ in range(3))
            if not(np.array_equal(a,b) and np.array_equal(a,d)): raise RuntimeError(f"{f['fold_id']}: deterministic failure")
            s=te[['trade_date','stock_code','target']].copy(); s['pred_prob']=a; s['fold_id']=f['fold_id']; s['label_mature_within_dev']=s.trade_date<=f['mature_label_cutoff']
            if s.duplicated(['trade_date','stock_code']).any() or s.stock_code.astype(str).str.endswith('.BJ').any() or s.pred_prob.isna().any() or not np.isfinite(s.pred_prob).all(): raise RuntimeError(f"{f['fold_id']}: quality failure")
            rows.append(s)
    finally: fd.close(); ld.close()
    cand=pd.concat(rows,ignore_index=True); cand.to_parquet(OUT/'candidate_oof.parquet',index=False)
    be=baseline.loc[baseline.label_mature_within_dev,['trade_date','stock_code','target','pred_prob','fold_id']].reset_index(drop=True); ce=cand.loc[cand.label_mature_within_dev,['trade_date','stock_code','target','pred_prob','fold_id']].reset_index(drop=True)
    if not be[['trade_date','stock_code']].equals(ce[['trade_date','stock_code']]): raise RuntimeError('same key failure')
    comparisons=[]
    for fid in ('fold2022','fold2023','fold2024'):
        before,after=metrics(be.query('fold_id==@fid')),metrics(ce.query('fold_id==@fid')); comparisons.append({'fold_id':fid,'baseline':before,'candidate':after,'delta':{k:after[k]-before[k] for k in before}})
    before,after=metrics(be),metrics(ce); delta={k:after[k]-before[k] for k in before}; gate={'aggregate_rank_ic_not_weaker':delta['rank_ic_mean']>=0,'each_fold_rank_ic_not_weaker':all(x['delta']['rank_ic_mean']>=0 for x in comparisons),'aggregate_top1_top3_top5_top10_not_weaker':all(delta[f'top{k}_excess_mean']>=0 for k in (1,3,5,10)),'each_fold_top10_not_weaker':all(x['delta']['top10_excess_mean']>=0 for x in comparisons)}
    summary={'candidate_id':c['candidate_id'],'stage':'development_2022_2024_only','same_key_rows':len(ce),'fold_comparisons':comparisons,'aggregate_baseline':before,'aggregate_candidate':after,'aggregate_delta':delta,'acceptance_gate':gate,'decision':'development_pass_waiting_for_2025_confirmation' if all(gate.values()) else 'reject_no_further_search','confirmation_2025_read':False,'validation_2026_closed':True,'production_unchanged':True,'allow_next_layer_continue':False}
    base.dump_json(OUT/'evaluation_summary.json',summary); base.dump_json(OUT/'audit_handoff.json',{'status':summary['decision'],'candidate_oof_sha256':base.sha256_file(OUT/'candidate_oof.parquet'),'summary_sha256':base.sha256_file(OUT/'evaluation_summary.json'),'contract_sha256':base.sha256_file(CONTRACT),'confirmation_2025_read':False,'allow_next_layer_continue':False,'production_unchanged':True})
    return 0
if __name__=='__main__': raise SystemExit(main())
