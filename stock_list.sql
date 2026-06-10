SELECT EXP(SUM(LN(RT+1))),avg(RT),avg(open2_yield_rate), count(1) FROM (
SELECT TRADE_DATE,AVG((sell_price-buy_price)/buy_price/1) AS RT, avg(open2_yield_rate+1) as open2_yield_rate,count(1) as cnt  FROM (
select *,
rank() over (partition by trade_date order by pred_prob DESC, atr_qfq/close desc) as rn from 
(
SELECT 
trade_date, name,
stock_code,pred_prob,
LEAD(pred_prob, 1) OVER (PARTITION BY stock_code ORDER BY trade_date) as post_pred_prob,
 open2_yield_rate,
 open3_yield_rate,
 industry_encode,
atr_qfq/close,
st_type,
 atr_qfq,
 limit_times,
 close,
close*1.04 as post_buy_price,post_open,post2_open,LEAD(post_open,1) OVER (PARTITION BY stock_code ORDER BY trade_date),
case 
when post_open <= close * 1.095 then post_open
else 0
end as buy_price,
case 
when post_open <= close * 1.095  then  post2_open
else 0
end  as sell_price
 FROM stock_predict_data_10d_yield_rate
 where  TRADE_DATE >= '20250601'
 ORDER BY TRADE_DATE DESC 
 ) AS T
 where  limit_times is null and st_type is null
 ) as T
 WHERE RN <= 10
	group by TRADE_DATE
 ORDER BY TRADE_DATE DESC, rn
 ) AS T
 
 1.68 1.51 1.39
 1.53 1.33 1.57

 
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select * from (
select *,
rank() over (partition by trade_date order by pred_prob DESC, atr_qfq/close desc) as rn from 
(
SELECT 
trade_date, name,
stock_code,pred_prob,
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
case 
when post_open <= close * 1.095 then post_open
else 0
end as buy_price,
case 
when post_open <= close * 1.095  then  post2_open
else 0
end  as sell_price,*
 FROM stock_predict_data_10d_yield_rate
 where  TRADE_DATE >= '2025-01-01'
 ORDER BY TRADE_DATE DESC 
 ) AS T
 where  limit_times is null and st_type is null
 order by TRADE_DATE desc ,rn
 ) where rn <= 500 or name in ('英可瑞','华如科技','荣旗科技')
 
 
 300522.SZ
600207.SH
SELECT * FROM STOCK_DAILY_DATA  WHERE STOCK_CODE = '600207.SH' LIMIT 100
SELECT * FROM DAILY_DATA  WHERE STOCK_CODE = '600207.SH' LIMIT 100