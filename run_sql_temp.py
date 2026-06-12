import sqlite3
import pandas as pd
from project_paths import resolve_data_dir

conn = sqlite3.connect(resolve_data_dir() / 'odb.db')
cursor = conn.cursor()

sql = '''
select * from (
 select *,
 rank() over (partition by trade_date order by sum_pred_prob DESC, atr_qfq/close desc) as rn from
 (
 SELECT
 trade_date, name,
 stock_code,pred_prob,
 1 * pred_prob + 1 * adjust_pred_code as sum_pred_prob,
 LEAD(pred_prob, 1) OVER (PARTITION BY stock_code ORDER BY trade_date) as post_pred_prob,
  open2_yield_rate,
  open3_yield_rate,
  industry_encode,
 atr_qfq/close,
 st_type,
  atr_qfq,
  limit_times,
 close*1.04 as post_buy_price,post_open,post2_open,LEAD(post_open,1) OVER (PARTITION BY stock_code ORDER BY trade_date),
 post_open,post2_open,close,
 post_open as buy_price,
 post2_open as sell_price,*
  FROM (
  select t1.*,t1.pred_prob as adjust_pred_code from
  stock_predict_data_10d_yield_rate as t1
  )
  where  TRADE_DATE >= '20250101'
  ORDER BY TRADE_DATE DESC
  ) AS T
 order by TRADE_DATE desc ,rn
  ) where rn <= 10 or name in ('中粮糖业', '欧派家居','国博电子','佰维存储','湖南裕能')
'''

cursor.execute(sql)
results = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]

df = pd.DataFrame(results, columns=columns)

latest_date = df['trade_date'].max()
print(f'最新交易日: {latest_date}')
print('='*80)

latest_stocks = df[df['trade_date'] == latest_date].sort_values('rn')

display_cols = ['trade_date', 'rn', 'name', 'stock_code', 'pred_prob', 'sum_pred_prob',
                'atr_qfq/close', 'buy_price', 'sell_price', 'open2_yield_rate', 'open3_yield_rate','rn']

print(latest_stocks[display_cols].to_string(index=False))

conn.close()
