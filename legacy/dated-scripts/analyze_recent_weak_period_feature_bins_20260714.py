from __future__ import annotations

import json
from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'quant' / 'data_file' / 'reports' / 'strategy_agent_four_year_l4_frequency_optimization_20260714'
MARKET_DB = ROOT / 'quant' / 'data_file' / 'production_assets' / 'duckdb' / 'l2_stock_daily_data.duckdb'
CASES = {
    'annual_boundary_ge_h69': REPORT_DIR / 'signals' / 'top3_pick1_gap_edge_refine' / 'ge_h69_m36_l27_ss80_pg35_ds100_dg110.csv',
    'sharpe_boundary_multi_t3': REPORT_DIR / 'signals' / 'top3_multi_open_quality_sizing' / 'multi_t3_h60_m30_l18_ss75_pg50.csv',
    'frontier_annual_mf_t2_sc1200': REPORT_DIR / 'signals' / 'top3_multi_open_quality_sizing' / 'mf_t2_sc1200_cap100.csv',
}


def load_market() -> pd.DataFrame:
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        return con.execute('''
            WITH cal AS (
                SELECT DISTINCT trade_date,
                       lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
                FROM STOCK_DAILY_DATA
            )
            SELECT m.trade_date, m.stock_code, m.open AS buy_open_raw,
                   n.open AS next_open_raw,
                   c.next_trade_date
            FROM STOCK_DAILY_DATA m
            JOIN cal c ON c.trade_date = m.trade_date
            LEFT JOIN STOCK_DAILY_DATA n
              ON n.trade_date = c.next_trade_date AND n.stock_code = m.stock_code
        ''').fetchdf()
    finally:
        con.close()


def bucket_pct(v: float) -> str:
    if pd.isna(v): return 'missing'
    if v <= -9: return '<=-9'
    if v <= -7: return '(-9,-7]'
    if v <= -5: return '(-7,-5]'
    if v <= -3: return '(-5,-3]'
    return '>-3'


def bucket_gap(v: float) -> str:
    if pd.isna(v): return 'missing'
    if v <= -5: return '<=-5'
    if v <= -3: return '(-5,-3]'
    if v <= -1: return '(-3,-1]'
    if v <= 0.5: return '(-1,0.5]'
    return '>0.5'


def bucket_pred1(v: float) -> str:
    if pd.isna(v): return 'missing'
    if v >= 0.9: return '>=0.9'
    if v >= 0.5: return '[0.5,0.9)'
    if v >= 0: return '[0,0.5)'
    return '<0'


def bucket_amount(v: float) -> str:
    if pd.isna(v): return 'missing'
    if v >= 500000: return '>=50w'
    if v >= 200000: return '20w-50w'
    if v >= 100000: return '10w-20w'
    return '<10w'


def bucket_mv(v: float) -> str:
    if pd.isna(v): return 'missing'
    if v >= 3000000: return '>=300w'
    if v >= 1000000: return '100w-300w'
    if v >= 500000: return '50w-100w'
    return '<50w'


def bucket_atr(v: float) -> str:
    if pd.isna(v): return 'missing'
    if v >= 6: return '>=6'
    if v >= 3: return '3-6'
    if v >= 1: return '1-3'
    return '<1'


def summarize(df: pd.DataFrame, col: str) -> list[dict]:
    out=[]
    for (year, bucket), g in df.groupby(['year', col], dropna=False):
        out.append({
            'year': str(year),
            'bucket': str(bucket),
            'rows': int(len(g)),
            'target_sum': float(g['target_pct'].sum()),
            'weighted_return_sum': float(g['weighted_return'].sum()),
            'mean_trade_return': float(g['trade_return'].mean()),
            'win_rate': float((g['trade_return'] > 0).mean()),
        })
    return out


def main():
    market = load_market()
    results = {}
    rows_all=[]
    for case, path in CASES.items():
        df = pd.read_csv(path, encoding='utf-8-sig', dtype={'buy_date':str,'signal_date':str,'stock_code':str})
        for col in ['target_pct','signal_pct_chg_raw','exec_open_gap_pct','buy_open_gap_pct','pred_1d','pred_10d','amount','total_mv','atr_qfq','quality_bucket']:
            if col in df.columns:
                df[col]=pd.to_numeric(df[col], errors='coerce')
        if 'exec_open_gap_pct' not in df.columns:
            df['exec_open_gap_pct']=pd.to_numeric(df.get('buy_open_gap_raw_pct', df.get('buy_open_gap_pct')), errors='coerce')
        merged=df.merge(market, left_on=['buy_date','stock_code'], right_on=['trade_date','stock_code'], how='left')
        merged['trade_return']=merged['next_open_raw']/merged['buy_open_raw']-1.0
        merged['weighted_return']=merged['trade_return']*merged['target_pct']
        merged['year']=merged['buy_date'].str[:4]
        merged['pct_bucket']=merged['signal_pct_chg_raw'].map(bucket_pct)
        merged['gap_bucket']=merged['exec_open_gap_pct'].map(bucket_gap)
        merged['pred1_bucket']=merged['pred_1d'].map(bucket_pred1)
        merged['amount_bucket']=merged['amount'].map(bucket_amount)
        merged['mv_bucket']=merged['total_mv'].map(bucket_mv)
        merged['atr_bucket']=merged['atr_qfq'].map(bucket_atr)
        merged['quality_bucket_label']=merged.get('quality_bucket', pd.Series([None]*len(merged))).fillna(-1).astype(int).astype(str)
        results[case]={
            'signal_file': str(path),
            'rows': int(len(merged)),
            'missing_market_rows': int(merged['buy_open_raw'].isna().sum()),
            'weighted_by_year': {str(k): float(v) for k,v in merged.groupby('year')['weighted_return'].sum().items()},
            'target_by_year': {str(k): float(v) for k,v in merged.groupby('year')['target_pct'].sum().items()},
            'bins': {
                'signal_pct_chg_raw': summarize(merged, 'pct_bucket'),
                'exec_open_gap_pct': summarize(merged, 'gap_bucket'),
                'pred_1d': summarize(merged, 'pred1_bucket'),
                'amount': summarize(merged, 'amount_bucket'),
                'total_mv': summarize(merged, 'mv_bucket'),
                'atr_qfq': summarize(merged, 'atr_bucket'),
                'quality_bucket': summarize(merged, 'quality_bucket_label'),
            },
        }
        tmp=merged.copy()
        tmp.insert(0,'case',case)
        rows_all.append(tmp)
    detail=pd.concat(rows_all, ignore_index=True)
    detail_path=REPORT_DIR/'recent_weak_period_feature_detail_20260714.csv'
    detail.to_csv(detail_path, index=False, encoding='utf-8-sig')
    out=REPORT_DIR/'recent_weak_period_feature_bins_20260714.json'
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    # compact table for main suspicious 2025/2026 negative/weak bins
    compact=[]
    for case,data in results.items():
        for feature,bins in data['bins'].items():
            for r in bins:
                if r['year'] in {'2025','2026'} and r['rows'] >= 5:
                    compact.append({'case':case,'feature':feature,**r})
    compact_df=pd.DataFrame(compact).sort_values(['year','weighted_return_sum'])
    compact_path=REPORT_DIR/'recent_weak_period_feature_bins_compact_20260714.csv'
    compact_df.to_csv(compact_path, index=False, encoding='utf-8-sig')
    print(json.dumps({'json':str(out),'detail':str(detail_path),'compact':str(compact_path)}, ensure_ascii=False, indent=2))

if __name__=='__main__':
    main()
