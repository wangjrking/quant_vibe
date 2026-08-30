import pandas as pd
import time
from datetime import datetime
import itertools
import json
import os
import tushare as ts
import logging
import multiprocessing
import sqlite3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
# 导入进度条工具
from tqdm import tqdm

from project_paths import load_config, resolve_data_dir
from l1_universe_rules import filter_frame_no_bj
from raw_table_db_module import replace_raw_table_full




def get_config():
	return load_config()


def _normalize_tushare_pro_endpoint(endpoint):
	if not endpoint:
		return None
	value = str(endpoint).strip().rstrip("/")
	if not value:
		return None
	if not value.endswith("/dataapi"):
		value = f"{value}/dataapi"
	return value


def _configured_tushare_pro_endpoint():
	for env_name in ("TUSHARE_PRO_API_URL", "TUSHARE_API_URL"):
		value = _normalize_tushare_pro_endpoint(os.environ.get(env_name))
		if value:
			return value
	try:
		config = load_config()
	except Exception:
		return None
	return _normalize_tushare_pro_endpoint(
		config.get("datasource", {}).get("tushare_pro_api_url")
	)


def configure_tushare_pro_endpoint(endpoint=None):
	value = _normalize_tushare_pro_endpoint(endpoint or _configured_tushare_pro_endpoint())
	if not value:
		return None
	from tushare.pro.client import DataApi
	DataApi._DataApi__http_url = value
	return value


def get_pro(ts_token, endpoint=None):
	configure_tushare_pro_endpoint(endpoint)
	ts_pro = ts.pro_api(ts_token)
	return ts_pro

def date_range_pandas(start_str, end_str):
    date_range = pd.date_range(
        start=pd.to_datetime(start_str, format='%Y%m%d'),
        end=pd.to_datetime(end_str, format='%Y%m%d'),
        freq='D'  # 按天生成
    )
    return [d.strftime('%Y%m%d') for d in date_range]

def optimize_dtypes(df):
    # 整数列优化
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # 浮点列优化
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # 对象/字符串列优化
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df[col] = df[col].astype('str')

    return df

def memory_usage(df):
    return df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB


# 获取行情数据
# 入参：stock_code_lst（股票代码列表）, data_start_dt（数据开始日期）, data_end_dt（数据结束日期）
# 出餐：data（DataFrame类型）--字段列表：stock_code, trade_date ... 各种指标
def get_daily_data(stock_code_lst, data_start_dt, data_end_dt, ts_pro):
	daily_data_dct = {}
	for stock_code in stock_code_lst:
		while True:

			try:
				daily_data_dct[stock_code] = ts_pro.query('daily', ts_code=stock_code, start_date=data_start_dt,end_date=data_end_dt)
				break
			except:
				logging.warning(f'调用失败，daily_basic接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	daily_data = pd.concat(daily_data_dct.values())
	daily_data = optimize_dtypes(daily_data)

	return daily_data



def get_daily_index_data(stock_code_lst, data_start_dt, data_end_dt, ts_pro):

	daily_index_data_dct = {}
	for stock_code in stock_code_lst:
		while True:
			try:
				daily_index_data_dct[stock_code] = ts_pro.query('daily_basic', ts_code=stock_code, start_date=data_start_dt,end_date=data_end_dt)
				break
			except:
				logging.warning(f'调用失败，daily_basic接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	daily_index_data = pd.concat(daily_index_data_dct.values())

	daily_index_data = optimize_dtypes(daily_index_data)

	return daily_index_data

def get_stock_basic_data(stock_code_lst, data_start_dt, data_end_dt, ts_pro):

	stock_basic_data = ts_pro.query('stock_basic')

	stock_basic_data = optimize_dtypes(stock_basic_data)

	return stock_basic_data

# 获取财务数据
# 入参：stock_code_lst（股票代码列表）, data_start_dt（数据开始日期）, data_end_dt（数据结束日期）, ts_pro（tushare查询引擎）
# 出餐：data（DataFrame类型）--字段列表：stock_code, ann_date ... 各种指标
def get_finan_data(stock_code_lst, data_start_dt, data_end_dt, ts_pro):
	finan_data_dct = {}
	for stock_code in stock_code_lst:
		while True:
			try:
				finan_data_dct[stock_code] = ts_pro.fina_indicator_vip(ts_code=stock_code, start_date=data_start_dt,end_date=data_end_dt)
				break
			except:
				logging.warning('调用失败，fina_indicator接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	finan_data = pd.concat(finan_data_dct.values())

	finan_data_season = finan_data[finan_data['end_date'].str.slice(4, 6) != '12']
	finan_data_season = finan_data_season[finan_data_season['update_flag']=='1']
	finan_data_season = finan_data_season.drop_duplicates(subset=['ts_code', 'ann_date'], keep='first')
	finan_data_season = finan_data_season.dropna(subset=['ts_code', 'ann_date'])


	finan_data_year = finan_data[finan_data['end_date'].str.slice(4, 6) == '12']
	finan_data_year = finan_data_year[finan_data_year['update_flag']=='1']
	finan_data_year = finan_data_year.drop_duplicates(subset=['ts_code', 'ann_date'], keep='first')
	finan_data_year = finan_data_year.dropna(subset=['ts_code', 'ann_date'])

	finan_data_season = optimize_dtypes(finan_data_season)
	finan_data_year = optimize_dtypes(finan_data_year)


	return finan_data_season, finan_data_year

def get_limit_list_data(stock_code_lst, data_start_dt, data_end_dt, ts_pro):

	limit_list_data_dct = {}
	for stock_code in stock_code_lst:
		while True:
			try:
				limit_list_data_dct[stock_code] = ts_pro.limit_list_d(ts_code=stock_code, limit_type='U',start_date=data_start_dt, end_date=data_end_dt)
				break
			except:
				logging.warning(f'调用失败，daily_basic接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	limit_list_data = pd.concat(limit_list_data_dct.values())


	limit_list_data = optimize_dtypes(limit_list_data)

	return limit_list_data

def get_adj_factor(stock_code_lst, data_start_dt, data_end_dt, ts_pro):


	adj_factor_dct = {}
	trade_date_lst = date_range_pandas(data_start_dt, data_end_dt)

	for trade_date in trade_date_lst:
		while True:
			try:
				adj_factor_dct[trade_date] = ts_pro.adj_factor(trade_date=trade_date)
				break
			except:
				logging.warning(f'调用失败，adj_factor接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	adj_factor = pd.concat(adj_factor_dct.values())


	adj_factor = optimize_dtypes(adj_factor)
	return adj_factor

def get_moneyflow(stock_code_lst, data_start_dt, data_end_dt, ts_pro):

	moneyflow_dct = {}
	for stock_code in stock_code_lst:
		while True:
			try:
				moneyflow_dct[stock_code] = ts_pro.moneyflow(ts_code=stock_code,start_date=data_start_dt, end_date=data_end_dt)
				break
			except:
				logging.warning(f'调用失败，moneyflow接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	moneyflow = pd.concat(moneyflow_dct.values())


	moneyflow = optimize_dtypes(moneyflow)
	return moneyflow

def get_stk_factor(stock_code_lst, data_start_dt, data_end_dt, ts_pro):

	stk_factor_dct = {}
	for stock_code in stock_code_lst:
		while True:
			try:
				stk_factor_dct[stock_code] = optimize_dtypes(ts_pro.stk_factor_pro(ts_code=stock_code,start_date=data_start_dt, end_date=data_end_dt))
				break
			except:
				logging.warning(f'调用失败，stk_factor_pro接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	stk_factor = pd.concat(stk_factor_dct.values())


	stk_factor = optimize_dtypes(stk_factor)
	return stk_factor



def get_tdx_index(data_start_dt, data_end_dt, ts_pro):

	tdx_index_dct = {}
	trade_date_lst = date_range_pandas(data_start_dt, data_end_dt)
	for trade_date in trade_date_lst:
		while True:
			try:
				tdx_index_dct[trade_date] = ts_pro.tdx_index(trade_date=trade_date,idx_type='行业板块')
				break
			except:
				logging.warning(f'调用失败，tdx_index接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	tdx_index = pd.concat(tdx_index_dct.values())


	tdx_index = optimize_dtypes(tdx_index)
	return tdx_index

def get_tdx_member(tdx_index_lst, ts_pro):

	tdx_member_dct = {}
	for ts_code,trade_date in tdx_index_lst:
		while True:
			try:
				tdx_member_dct[ts_code+trade_date] = ts_pro.tdx_member(ts_code=ts_code,trade_date=trade_date)
				break
			except:
				logging.warning(f'调用失败，tdx_member接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	tdx_member = pd.concat(tdx_member_dct.values())


	tdx_member = optimize_dtypes(tdx_member)
	return tdx_member

def get_tdx_daily(data_start_dt, data_end_dt, ts_pro):

	tdx_daily_dct = {}
	trade_date_lst = date_range_pandas(data_start_dt, data_end_dt)
	for trade_date in trade_date_lst:
		while True:
			try:
				tdx_daily_dct[trade_date] = ts_pro.tdx_daily(trade_date=trade_date)
				break
			except:
				logging.warning(f'调用失败，tdx_index接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	tdx_daily = pd.concat(tdx_daily_dct.values())


	tdx_daily = optimize_dtypes(tdx_daily)
	return tdx_daily

# 获取龙虎榜数据
def get_top_list(data_start_dt, data_end_dt, ts_pro):

	top_list_dct = {}
	trade_date_lst = date_range_pandas(data_start_dt, data_end_dt)
	for trade_date in trade_date_lst:
		while True:
			try:
				top_list_dct[trade_date] = ts_pro.top_list(trade_date=trade_date)
				break
			except:
				logging.warning(f'调用失败，top_list接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	top_list = pd.concat(top_list_dct.values())


	top_list = optimize_dtypes(top_list)

	top_list = top_list.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
	return top_list

# 获取同花顺热榜
def get_ths_hot(data_start_dt, data_end_dt, ts_pro):

	ths_hot_dct = {}
	trade_date_lst = date_range_pandas(data_start_dt, data_end_dt)
	for trade_date in trade_date_lst:
		while True:
			try:
				ths_hot_dct[trade_date] = ts_pro.ths_hot(trade_date=trade_date, market='热股')
				break
			except:
				logging.warning(f'调用失败，ths_hot接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	ths_hot = pd.concat(ths_hot_dct.values())


	ths_hot = optimize_dtypes(ths_hot)

	ths_hot = ths_hot.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
	return ths_hot

# 获取东方财富热榜
def get_dc_hot(data_start_dt, data_end_dt, ts_pro):

	dc_hot_dct = {}
	trade_date_lst = date_range_pandas(data_start_dt, data_end_dt)
	for trade_date in trade_date_lst:
		while True:
			try:
				dc_hot_dct[trade_date] = ts_pro.dc_hot(trade_date=trade_date,market='A股市场',hot_type='人气榜')
				break
			except:
				logging.warning(f'调用失败，dc_hot接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	dc_hot = pd.concat(dc_hot_dct.values())


	dc_hot = optimize_dtypes(dc_hot)

	dc_hot = dc_hot.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
	return dc_hot

# 获取筹码盘分布
def get_cyq_perf(stock_code_lst, data_start_dt, data_end_dt, ts_pro):

	cyq_perf_dct = {}
	for stock_code in stock_code_lst:
		while True:
			try:
				cyq_perf_dct[stock_code] = ts_pro.cyq_perf(ts_code=stock_code,start_date=data_start_dt, end_date=data_end_dt)
				break
			except:
				logging.warning(f'调用失败，cyq_perf接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	cyq_perf = pd.concat(cyq_perf_dct.values())


	cyq_perf = optimize_dtypes(cyq_perf)
	return cyq_perf

# 获取ST股票列表
def get_stock_st(data_start_dt, data_end_dt, ts_pro):

	stock_st_dct = {}
	trade_date_lst = date_range_pandas(data_start_dt, data_end_dt)
	for trade_date in trade_date_lst:
		while True:
			try:
				stock_st_dct[trade_date] = ts_pro.stock_st(trade_date=trade_date)
				break
			except:
				logging.warning(f'调用失败，stock_st接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	stock_st = pd.concat(stock_st_dct.values())


	stock_st = optimize_dtypes(stock_st)

	stock_st = stock_st.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
	return stock_st


def get_index_daily(data_start_dt, data_end_dt, ts_pro, stock_code_lst=None):

	if stock_code_lst is None:
		stock_code_lst = ['000300.SH', '000905.SH', '932000.CSI']
	index_daily_dct = {}
	for stock_code in stock_code_lst:
		while True:
			try:
				index_daily_dct[stock_code] = ts_pro.query('index_daily', ts_code=stock_code, start_date=data_start_dt,end_date=data_end_dt)
				break
			except:
				logging.warning(f'调用失败，index_daily接口调用次数过多，需要等待1分钟')
				time.sleep(61)

	index_daily = pd.concat(index_daily_dct.values())

	index_daily = optimize_dtypes(index_daily)

	return index_daily

def store_daily_data(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 获取日行情数据
	daily_data = filter_frame_no_bj(get_daily_data(stock_code_lst, data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：2.【日行情数据】读取完成--daily_data')
	# daily_data.to_sql('daily_data', con=conn, if_exists='replace', index=False)
	daily_data.to_parquet(data_file_url + '/daily_data.parquet',index=False)
	logging.info(f'数据接入模块：2.数据表导出成功 库名：odb，表名：daily_data')
	return ['daily_data']



def store_daily_index_data(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 获取每日基本面指标数据
	daily_index_data = filter_frame_no_bj(get_daily_index_data(stock_code_lst, data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：3.【基本面指标】读取完成--daily_index_data')
	# daily_index_data.to_sql('daily_index_data', con=conn, if_exists='replace', index=False)
	daily_index_data.to_parquet(data_file_url + '/daily_index_data.parquet',index=False)
	logging.info(f'数据接入模块：3.数据表导出成功 库名：odb，表名：daily_index_data')
	return ['daily_index_data']



def store_stock_basic_data(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 获取股票基本信息
	stock_basic_data = filter_frame_no_bj(get_stock_basic_data(stock_code_lst, data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：4.【股票基本信息】读取完成--stock_basic_data')
	# stock_basic_data.to_sql('stock_basic_data', con=conn, if_exists='replace', index=False)
	stock_basic_data.to_parquet(data_file_url + '/stock_basic_data.parquet',index=False)
	logging.info(f'数据接入模块：4.数据表导出成功 库名：odb，表名：stock_basic_data')
	return ['stock_basic_data']



def store_finan_data(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 获取股票的财务数据
	finan_data_season, finan_data_year = get_finan_data(stock_code_lst, data_start_dt, data_end_dt, ts_pro)
	logging.info(f'数据接入模块：5.【财务季度报指标】读取完成--finan_data_season')
	# finan_data_season.to_sql('finan_data_season', con=conn, if_exists='replace', index=False)
	finan_data_season.to_parquet(data_file_url + '/finan_data_season.parquet',index=False)
	logging.info(f'数据接入模块：5.数据表导出成功 库名：odb，表名：finan_data_season')
	logging.info(f'数据接入模块：6.【财务年报指标】读取完成--finan_data_year')
	# finan_data_year.to_sql('finan_data_year', con=conn, if_exists='replace', index=False)
	finan_data_year.to_parquet(data_file_url + '/finan_data_year.parquet',index=False)
	logging.info(f'数据接入模块：6.数据表导出成功 库名：odb，表名：finan_data_year')
	return ['finan_data_year','finan_data_season']



def store_limit_list_data(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 获取打板数据
	limit_list_data = filter_frame_no_bj(get_limit_list_data(stock_code_lst, data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：7.【股票打板指标】读取完成--limit_list_data')
	# limit_list_data.to_sql('limit_list_data', con=conn, if_exists='replace', index=False)
	limit_list_data.to_parquet(data_file_url + '/limit_list_data.parquet',index=False)
	logging.info(f'数据接入模块：7.数据表导出成功 库名：odb，表名：limit_list_data')
	return ['limit_list_data']



def store_adj_factor(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 获取复权因子
	adj_factor = filter_frame_no_bj(get_adj_factor(stock_code_lst, data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：8.【复权因子】读取完成--adj_factor')
	# adj_factor.to_sql('adj_factor', con=conn, if_exists='replace', index=False)
	adj_factor.to_parquet(data_file_url + '/adj_factor.parquet',index=False)
	logging.info(f'数据接入模块：8.数据表导出成功 库名：odb，表名：adj_factor')
	return ['adj_factor']



def store_moneyflow(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 获取资金流向
	moneyflow = filter_frame_no_bj(get_moneyflow(stock_code_lst, data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：9.【资金流向】读取完成--moneyflow')
	# moneyflow.to_sql('moneyflow', con=conn, if_exists='replace', index=False)
	moneyflow.to_parquet(data_file_url + '/moneyflow.parquet',index=False)
	logging.info(f'数据接入模块：9.数据表导出成功 库名：odb，表名：moneyflow')
	return ['moneyflow']



def store_stk_factor(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 获取股票技术面因子
	stk_factor = filter_frame_no_bj(get_stk_factor(stock_code_lst, data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：10.【股票技术面因子】读取完成--stk_factor')
	# stk_factor.to_sql('stk_factor', con=conn, if_exists='replace', index=False)
	stk_factor.to_parquet(data_file_url + '/stk_factor.parquet',index=False)
	logging.info(f'数据接入模块：10.数据表导出成功 库名：odb，表名：stk_factor')
	return ['stk_factor']





def store_top_list(data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 龙虎榜数据
	top_list = filter_frame_no_bj(get_top_list(data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：14.【龙虎榜数据】读取完成--top_list')
	# top_list.to_sql('top_list', con=conn, if_exists='replace', index=False)
	top_list.to_parquet(data_file_url + '/top_list.parquet',index=False)
	logging.info(f'数据接入模块：14.数据表导出成功 库名：odb，表名：top_list')
	return ['top_list']



def store_ths_hot(data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 同花顺数据
	ths_hot = filter_frame_no_bj(get_ths_hot(data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：15.【同花顺热度】读取完成--ths_hot')
	# ths_hot.to_sql('ths_hot', con=conn, if_exists='replace', index=False)
	ths_hot.to_parquet(data_file_url + '/ths_hot.parquet',index=False)
	logging.info(f'数据接入模块：15.数据表导出成功 库名：odb，表名：ths_hot')
	return ['ths_hot']



def store_dc_hot(data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 东方财富热度
	dc_hot = filter_frame_no_bj(get_dc_hot(data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：16.【东方财富热度】读取完成--dc_hot')
	# dc_hot.to_sql('dc_hot', con=conn, if_exists='replace', index=False)
	dc_hot.to_parquet(data_file_url + '/dc_hot.parquet',index=False)
	logging.info(f'数据接入模块：16.数据表导出成功 库名：odb，表名：dc_hot')
	return ['dc_hot']



def store_cyq_perf(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 筹码盘分布
	cyq_perf = filter_frame_no_bj(get_cyq_perf(stock_code_lst, data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：17.【筹码盘分布】读取完成--cyq_perf')
	# cyq_perf.to_sql('cyq_perf', con=conn, if_exists='replace', index=False)
	cyq_perf.to_parquet(data_file_url + '/cyq_perf.parquet',index=False)
	logging.info(f'数据接入模块：17.数据表导出成功 库名：odb，表名：cyq_perf')
	# print(cyq_perf)
	return ['cyq_perf']



def store_stock_st(data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# ST股票列表
	stock_st = filter_frame_no_bj(get_stock_st(data_start_dt, data_end_dt, ts_pro))
	logging.info(f'数据接入模块：18.【ST股票列表】读取完成--stock_st')
	# stock_st.to_sql('stock_st', con=conn, if_exists='replace', index=False)
	stock_st.to_parquet(data_file_url + '/stock_st.parquet',index=False)
	logging.info(f'数据接入模块：18.数据表导出成功 库名：odb，表名：stock_st')
	return ['stock_st']

def store_index_daily(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):
	ts_pro = get_pro(ts_token)
	# 获取日行情数据
	index_daily = get_index_daily(data_start_dt, data_end_dt, ts_pro)
	logging.info(f'数据接入模块：19.【日行情数据】读取完成--index_daily')
	# daily_data.to_sql('daily_data', con=conn, if_exists='replace', index=False)
	index_daily.to_parquet(data_file_url + '/index_daily.parquet',index=False)
	logging.info(f'数据接入模块：19.数据表导出成功 库名：odb，表名：index_daily')
	return ['index_daily']


def download_odb_data(stock_code_lst, data_start_dt, data_end_dt, ts_token, data_file_url):

	logging.info(f'#--------------------------------------------数据接入模块启动--------------------------------------------#')
	logging.info(f'股票数量：{str(len(stock_code_lst))}')
	logging.info(f'开始日期：{data_start_dt}')
	logging.info(f'结束日期：{data_end_dt}')


	logging.info(f'数据接入模块：1.MYSQL数据库连接成功')


	# mutil_process1(1,stock_code_lst,data_start_dt, data_end_dt, ts_pro, engine)
	with ProcessPoolExecutor() as executor:
		futures = [
            executor.submit(store_daily_data,stock_code_lst,  data_start_dt, data_end_dt, ts_token, data_file_url),
            executor.submit(store_daily_index_data,stock_code_lst,  data_start_dt, data_end_dt, ts_token, data_file_url),
            executor.submit(store_stock_basic_data,stock_code_lst,  data_start_dt, data_end_dt, ts_token, data_file_url),
            executor.submit(store_finan_data,stock_code_lst,  data_start_dt, data_end_dt, ts_token, data_file_url),
			executor.submit(store_limit_list_data,stock_code_lst,  data_start_dt, data_end_dt, ts_token,data_file_url),
			executor.submit(store_adj_factor,stock_code_lst,  data_start_dt, data_end_dt, ts_token, data_file_url),
			executor.submit(store_moneyflow,stock_code_lst,  data_start_dt, data_end_dt, ts_token, data_file_url),
			executor.submit(store_stk_factor,stock_code_lst,  data_start_dt, data_end_dt, ts_token, data_file_url),
			executor.submit(store_cyq_perf,stock_code_lst,  data_start_dt, data_end_dt, ts_token, data_file_url),
			executor.submit(store_top_list, data_start_dt, data_end_dt, ts_token, data_file_url),
			executor.submit(store_ths_hot, data_start_dt, data_end_dt, ts_token, data_file_url),
			executor.submit(store_dc_hot, data_start_dt, data_end_dt, ts_token, data_file_url),
			executor.submit(store_stock_st, data_start_dt, data_end_dt, ts_token, data_file_url),
			executor.submit(store_index_daily,stock_code_lst,  data_start_dt, data_end_dt, ts_token, data_file_url),
        ]
		output_data = []
        # 👇 这里加 desc="你想要的名字" 就行
		for f in tqdm(futures, desc="多个数据接口任务读取中"):
			result = f.result()
			output_data +=  result

		for db in tqdm(output_data, desc="多个数据任务录入数据库中"):
			# print(key, value)
			data = pd.read_parquet(data_file_url + '/'+ db +'.parquet')
			replace_raw_table_full(data_file_url, db, data)


	'''
	# 创建进程实例

	p1 = multiprocessing.Process(target=mutil_process1, args=(stock_code_lst,data_start_dt, data_end_dt, ts_token))
	p2 = multiprocessing.Process(target=mutil_process2, args=(stock_code_lst,data_start_dt, data_end_dt, ts_token))
	p3 = multiprocessing.Process(target=mutil_process3, args=(stock_code_lst,data_start_dt, data_end_dt, ts_token))

    # 启动进程
	p1.start()
	logging.info(f'数据接入模块：进程1启动')
	p2.start()
	logging.info(f'数据接入模块：进程2启动')
	p3.start()
	logging.info(f'数据接入模块：进程3启动')

    # 等待子进程执行完毕（主进程阻塞）
	p1.join()
	logging.info(f'数据接入模块：进程1完成')
	p2.join()
	logging.info(f'数据接入模块：进程2完成')
	p3.join()
	logging.info(f'数据接入模块：进程3完成')
	'''

	logging.info(f'数据接入模块：数据表入仓完成')

	logging.info(f'#--------------------------------------------数据接入模块完成--------------------------------------------#')


if __name__ == '__main__':
    raise SystemExit("Use run_all_a_raw_update.py or a dedicated backfill script for data ingestion.")
