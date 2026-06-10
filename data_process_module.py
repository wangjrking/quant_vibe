# 导入数据分析库
import pandas as pd
# 导入技术分析库
import ta
from ta_compat import dema, kama, sma, midpoint, midprice, t3, tema, trima, wma, ema, rsi, atr, cmf, cci
# 导入数据编码工具
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
# 导入数值计算库
import numpy as np
# 导入进度条工具
from tqdm import tqdm
# 导入操作系统模块
import os
# 导入日志模块
import logging
from pathlib import Path

import sqlite3



	
# 数据整合
def get_integ_data(data_file_url):
	conn = sqlite3.connect(data_file_url + "/odb.db")
	integ_data = pd.read_sql("SELECT * FROM STOCK_DAILY_DATA order by stock_code, trade_date", conn)
	# integ_data = conn.execute("SELECT * FROM STOCK_DAILY_DATA order by stock_code, trade_date").df()
	conn.close()    
	# 记录日志
	logging.info(f"integ_data shape={integ_data.shape}")

	# 将列名转换为小写
	integ_data.columns = integ_data.columns.str.lower()

	# 返回整合数据
	return integ_data




def group_factor_eng(group_data):
	"""对单个股票数据进行因子工程

	Args:
		group_data: 单个股票的历史数据
		
	Returns:
		pd.DataFrame: 包含因子的股票数据
	"""
	# 存储新列的字典
	new_columns = {}
	
	# 技术指标
	
	# 计算MACD指标
	try:
		macd_indicator = ta.trend.MACD(group_data['close'], window_fast=12, window_slow=26, window_sign=9)
		new_columns['macd'] = macd_indicator.macd()
		new_columns['macdsignal'] = macd_indicator.macd_signal()
		new_columns['macdhist'] = macd_indicator.macd_diff()
	except:
		# 如果计算失败，填充为NaN
		new_columns['macd'] = np.nan
		new_columns['macdsignal'] = np.nan
		new_columns['macdhist'] = np.nan
	
	# 计算各种移动平均线指标
	new_columns['dema'] = dema(group_data['close'], length=30)  # 双指数移动平均线
	new_columns['ema'] = ema(group_data['close'], length=30)  # 指数移动平均线
	new_columns['kama'] = kama(group_data['close'], length=30)  # 自适应移动平均线
	new_columns['dema_10'] = dema(group_data['close'], length=10)  # 10周期双指数移动平均线
	new_columns['ema_10'] = ema(group_data['close'], length=10)  # 10周期指数移动平均线
	new_columns['kama_10'] = kama(group_data['close'], length=10)  # 10周期自适应移动平均线
	new_columns['ma'] = sma(group_data['close'], length=30)  # 简单移动平均线
	new_columns['midpoint'] = midpoint(group_data['close'], length=14)  # 中点价
	new_columns['midprice'] = midprice(group_data['high'], group_data['low'], length=14)  # 中间价
	new_columns['t3'] = t3(group_data['close'], length=5)  # T3移动平均线
	new_columns['tema'] = tema(group_data['close'], length=30)  # 三重指数移动平均线
	new_columns['trima'] = trima(group_data['close'], length=30)  # 三角移动平均线
	new_columns['wma'] = wma(group_data['close'], length=30)  # 加权移动平均线
	
	# 计算其他技术指标
	new_columns['rsi'] = rsi(group_data['close'], length=14)  # 相对强弱指标
	new_columns['atr'] = atr(group_data['high'], group_data['low'], group_data['close'], length=14)  # 平均真实波动范围
	new_columns['cmf'] = cmf(group_data['high'], group_data['low'], group_data['close'], group_data['vol'], length=14)  # 资金流量指标
	new_columns['cci'] = cci(group_data['high'], group_data['low'], group_data['close'], length=14)  # 顺势指标

	# 计算价格差异指标
	new_columns['diff_close_low'] = group_data['close'] - group_data['low']  # 收盘价与最低价的差
	new_columns['diff_close_high'] = group_data['close'] - group_data['high']  # 收盘价与最高价的差
	new_columns['diff_high_low'] = group_data['high'] - group_data['low']  # 最高价与最低价的差

	# 计算价格差异的指数移动平均线
	new_columns['ema_close_low'] = ema(new_columns['diff_close_low'], length=30)
	new_columns['ema_close_high'] = ema(new_columns['diff_close_high'], length=30)
	new_columns['ema_high_low'] = ema(new_columns['diff_high_low'], length=30)
	new_columns['ema_close_low_10'] = ema(new_columns['diff_close_low'], length=10)
	new_columns['ema_close_high_10'] = ema(new_columns['diff_close_high'], length=10)
	new_columns['ema_high_low_10'] = ema(new_columns['diff_high_low'], length=10)
	
	# 筹码盘分布清洗
	new_columns['std_his_low'] = group_data['his_low'] / group_data['close']  # 历史最低价标准化
	new_columns['std_his_high'] = group_data['his_high'] / group_data['close']  # 历史最高价标准化
	new_columns['std_cost_5pct'] = group_data['cost_5pct'] / group_data['close']  # 5%成本价标准化
	new_columns['std_cost_15pct'] = group_data['cost_15pct'] / group_data['close']  # 15%成本价标准化
	new_columns['std_cost_50pct'] = group_data['cost_50pct'] / group_data['close']  # 50%成本价标准化
	new_columns['std_cost_85pct'] = group_data['cost_85pct'] / group_data['close']  # 85%成本价标准化
	new_columns['std_cost_95pct'] = group_data['cost_95pct'] / group_data['close']  # 95%成本价标准化
	new_columns['std_weight_avg'] = group_data['weight_avg'] / group_data['close']  # 加权平均成本标准化

	# 打板因子
	new_columns['fd_amount_rate'] = group_data['fd_amount']/group_data['amount']  # 封单金额占比
	new_columns['fd_vol_rate'] = (group_data['fd_amount']/group_data['close'])/group_data['vol']  # 封单量占比
	new_columns['first_time_int'] = pd.to_numeric(group_data['first_time'], errors='coerce').astype('Int64')  # 首次涨停时间
	new_columns['last_time_int'] = pd.to_numeric(group_data['last_time'], errors='coerce').astype('Int64')  # 最后涨停时间
	
	# 分解涨跌统计数据
	up_stat_split = group_data['up_stat'].astype("string").fillna('0/0').str.split('/', expand=True)
	new_columns['up_stat_nom'] = up_stat_split[0]
	new_columns['up_stat_denom'] = up_stat_split[1]

	# 当日走势因子
	# new_columns['pre_close'] = group_data['close'].shift(1)  # 前一天收盘价

	# 计算价格相对于前收盘价的比率
	group_data = group_data.drop("pre_close", axis=1)
	group_data['pre_close'] = group_data['close'].shift(1) # 昨天收盘价
	new_columns['close_rate'] = group_data['close'] / group_data['pre_close']  # 收盘价比率
	new_columns['open_rate'] = group_data['open'] / group_data['pre_close']  # 开盘价比率
	new_columns['high_rate'] = group_data['high'] / group_data['pre_close']  # 最高价比率
	new_columns['low_rate'] = group_data['low'] / group_data['pre_close']  # 最低价比率

	# 计算价格间的比率差异
	new_columns['high_open_rate'] = new_columns['high_rate'] - new_columns['open_rate']  # 最高价与开盘价比率差
	new_columns['high_close_rate'] = new_columns['high_rate'] - new_columns['close_rate']  # 最高价与收盘价比率差
	new_columns['low_open_rate'] = new_columns['low_rate'] - new_columns['open_rate']  # 最低价与开盘价比率差
	new_columns['low_close_rate'] = new_columns['low_rate'] - new_columns['close_rate']  # 最低价与收盘价比率差

	# 转换为数值类型
	new_columns['up_stat_nom'] = new_columns['up_stat_nom'].astype(float)  # 涨跌统计分子
	new_columns['up_stat_denom'] = new_columns['up_stat_denom'].astype(float)  # 涨跌统计分母
	# 计算比例
	new_columns['up_stat_rate'] = new_columns['up_stat_nom'] / new_columns['up_stat_denom']  # 涨跌统计比率

	# 基本面因子

	# 计算ST标识：1表示ST，2表示*ST，0表示非ST
	new_columns['st_code'] = (
	group_data['name']
	.str[:2].eq('ST').astype(int) * 1 +          # 前2字符为ST → 1
	group_data['name']
	.str[:3].eq('*ST').astype(int) * 2 + 0           # 前3字符为*ST → 2
)

	# 时间因子
	data_dt = pd.to_datetime(group_data['trade_date'], format='%Y%m%d')  # 转换交易日期为 datetime 类型
	new_columns['year'] = data_dt.dt.year  # 年份
	new_columns['month'] = data_dt.dt.month  # 月份
	new_columns['quarter'] = data_dt.dt.quarter  # 季度
	new_columns['week_of_year'] = data_dt.dt.isocalendar().week  # 当年第几周
	new_columns['weekday'] = data_dt.dt.weekday  # 星期几（0-6，周一到周日）

	# 计算距离基准日期的天数
	base_date = pd.to_datetime('20200101', format='%Y%m%d')
	new_columns['days'] = (data_dt - base_date).dt.days
	new_columns['days_from_20200101'] = (data_dt - base_date).dt.days
	
	# 收益指标
	new_columns['pre_yield_rate'] = (group_data['close'] - group_data['pre_close'])/group_data['pre_close'] # 前一天收益率
	
	# 未来函数（用于预测）
	new_columns['post_close'] = group_data['close'].shift(-1) # 后一天收盘价
	new_columns['post2_close'] = group_data['close'].shift(-2) # 后两天收盘价
	new_columns['post5_close'] = group_data['close'].shift(-5) # 后五天收盘价
	new_columns['post6_close'] = group_data['close'].shift(-6) # 后六天收盘价
	new_columns['post10_close'] = group_data['close'].shift(-10) # 后十天收盘价
	new_columns['post15_close'] = group_data['close'].shift(-15) # 后15天收盘价
	new_columns['post22_close'] = group_data['close'].shift(-22) # 后一月收盘价
	new_columns['post_low'] = group_data['low'].shift(-1) # 后一天最低价
	new_columns['post_high'] = group_data['high'].shift(-1) # 后一天最高价
	new_columns['post_open'] = group_data['open'].shift(-1) # 后一天开盘价
	new_columns['post2_open'] = group_data['open'].shift(-2) # 后两天开盘价
	new_columns['post3_open'] = group_data['open'].shift(-3) # 后3天开盘价
	new_columns['post4_open'] = group_data['open'].shift(-4) # 后4天开盘价
	new_columns['post5_open'] = group_data['open'].shift(-5) # 后5天开盘价
	new_columns['post6_open'] = group_data['open'].shift(-6) # 后6天开盘价
	new_columns['post12_open'] = group_data['open'].shift(-12) # 后12天开盘价
	new_columns['post22_open'] = group_data['open'].shift(-22) # 后22天开盘价
	new_columns['post2_high'] = group_data['high'].shift(-2) # 后2天最高价
	new_columns['post3_high'] = group_data['high'].shift(-3) # 后3天最高价
	new_columns['post4_high'] = group_data['high'].shift(-4) # 后4天最高价
	new_columns['post5_high'] = group_data['high'].shift(-5) # 后5天最高价
 
	
	new_columns['index_2000_post10_close'] = group_data['index_2000_close'].shift(-10) # 后十天中证2000收盘价
	
	# 计算后5天和后2天最高值
	post_highs = pd.DataFrame({
		'post_high': new_columns['post_high'],
		'post2_high': new_columns['post2_high'],
		'post3_high': new_columns['post3_high'],
		'post4_high': new_columns['post4_high'],
		'post5_high': new_columns['post5_high']
	}, index=group_data.index)
	new_columns['post5_most_high'] = post_highs.max(axis=1) # 后5天最高值
	new_columns['post2_most_high'] = post_highs[['post_high', 'post2_high']].max(axis=1) # 后2天最高值
	
	# 计算涨跌标签
	new_columns['tag'] = ((new_columns['post_close'] - group_data['close']) > 0).astype(int) # 后一天涨跌标签
	new_columns['5d_tag'] = ((new_columns['post5_close'] - group_data['close']) > 0).astype(int) # 后5天涨跌标签
	
	# 计算各种收益率指标
	new_columns['yield_rate'] = (new_columns['post_close'] - group_data['close'])/group_data['close'] # 后一天收益率
	new_columns['close_yield_rate'] = (new_columns['post2_close'] - new_columns['post_close'])/new_columns['post_close'] # 后两天收盘价收益率
	
	new_columns['open_yield_rate'] = (new_columns['post_open'] - group_data['close'])/group_data['close'] # 后一天开盘价收益率
	new_columns['open2_yield_rate'] = (new_columns['post2_open'] - new_columns['post_open'])/new_columns['post_open'] # 后两天开盘价收益率
	new_columns['open3_yield_rate'] = (new_columns['post3_open'] - new_columns['post_open'])/new_columns['post_open'] # 后3天开盘价收益率
	new_columns['open4_yield_rate'] = (new_columns['post4_open'] - new_columns['post_open'])/new_columns['post_open'] # 后4天开盘价收益率
	new_columns['open5_yield_rate'] = (new_columns['post5_open'] - new_columns['post_open'])/new_columns['post_open'] # 后5天开盘价收益率
	new_columns['open6_yield_rate'] = (new_columns['post6_open'] - new_columns['post_open'])/new_columns['post_open'] # 后6天开盘价收益率
	new_columns['open12_yield_rate'] = (new_columns['post12_open'] - new_columns['post_open'])/new_columns['post_open'] # 后12天开盘价收益率
	new_columns['open22_yield_rate'] = (new_columns['post22_open'] - new_columns['post_open'])/new_columns['post_open'] # 后22天开盘价收益率
	new_columns['5d_yield_rate'] = (new_columns['post5_close'] - group_data['close'])/group_data['close'] # 后5天收盘价收益率
	new_columns['6d_yield_rate'] = (new_columns['post6_close'] - group_data['close'])/group_data['close'] # 后6天收盘价收益率
	new_columns['10d_yield_rate'] = (new_columns['post10_close'] - group_data['close'])/group_data['close'] # 后10天收盘价收益率
	new_columns['15d_yield_rate'] = (new_columns['post15_close'] - group_data['close'])/group_data['close'] # 后15天收盘价收益率
	new_columns['22d_yield_rate'] = (new_columns['post22_close'] - group_data['close'])/group_data['close'] # 后22天收盘价收益率
	new_columns['log_yield_rate'] = np.log(new_columns['yield_rate']+1) # 对数收益率
	new_columns['post5_most_high_yield_rate'] = (new_columns['post5_most_high'] - new_columns['post_open'])/new_columns['post_open'] # 后5天最高值相对于开盘价的收益率
 
 
	new_columns['index_2000_post10_yield_rate'] = (new_columns['index_2000_post10_close'] - group_data['index_2000_close'])/group_data['index_2000_close'] # 后10天中证2000收益率
 
	new_columns['adjust_10d_yield_rate'] = new_columns['10d_yield_rate'] - new_columns['index_2000_post10_yield_rate'] # 调整后的10天收益率

	# 计算2天收益率相关指标
	new_columns['2d_yield_rate'] = (new_columns['post2_close'] - group_data['close'])/group_data['close'] # 后2天收盘价收益率
	new_columns['2d_tag'] = (new_columns['2d_yield_rate'] > 0).astype(int) # 后2天涨跌标签
	new_columns['log_2d_yield_rate'] = np.log(new_columns['2d_yield_rate']+1) # 2天对数收益率

	# 计算其他收益率指标
	new_columns['next_open_yield_rate'] = (new_columns['post_open'] - group_data['close'])/group_data['close'] # 后一天开盘价相对当日收盘价收益率
	new_columns['close_open_yield_rate'] = (new_columns['post_close'] - new_columns['post_open'])/new_columns['post_open'] # 后一天收盘价相对于开盘价的收益率
	new_columns['close_low_yield_rate'] = (new_columns['post_close'] - new_columns['post_low'])/new_columns['post_low'] # 后一天收盘价相对于最低价的收益率
	new_columns['low_close_yield_rate'] = (
	(group_data['close'] - new_columns['post_low']).where(
		group_data['close'] < new_columns['post_low'],  # 条件：差值为负
		0  # 正差时替换为0
	) / group_data['close']
	) # 计算特定条件下的收益率
	
	# 处理缺失值
	new_columns['tag'] = new_columns['tag'].where(new_columns['post_close'].notna(), np.nan)
	
	# ==================== 国泰君安191因子库 ====================
	
	# 辅助函数：Count函数（计算满足条件的天数）
	def count_func(condition, window):
		return condition.rolling(window).sum()
	
	# 辅助函数：计算时间序列排名
	def ts_rank(series, window):
		return series.rolling(window).rank()
	
	# 辅助函数：计算回归斜率
	def regbeta(y, x, window):
		def beta(window_data):
			if len(window_data) < 2:
				return np.nan
			y_vals = window_data[:, 0]
			x_vals = window_data[:, 1]
			return np.cov(y_vals, x_vals)[0, 1] / np.var(x_vals)
		return pd.concat([y, x], axis=1).rolling(window).apply(beta, raw=True)
	
	# 辅助函数：计算回归截距
	def regresi(y, x, window):
		def intercept(window_data):
			if len(window_data) < 2:
				return np.nan
			y_vals = window_data[:, 0]
			x_vals = window_data[:, 1]
			beta_val = np.cov(y_vals, x_vals)[0, 1] / np.var(x_vals)
			return y_vals.mean() - beta_val * x_vals.mean()
		return pd.concat([y, x], axis=1).rolling(window).apply(intercept, raw=True)
	
	# 计算VWAP（成交量加权平均价）
	vwap = (group_data['open'] + group_data['high'] + group_data['low'] + group_data['close']) / 4
	
	# 计算ADV（平均成交量）
	adv10 = group_data['vol'].rolling(10).mean()
	adv20 = group_data['vol'].rolling(20).mean()
	
	# ==================== 国泰君安191因子库 ====================
	# 注意：Alpha191因子中的RANK()是截面排名（cross-sectional rank）
	# 需要在合并所有股票数据后，按交易日进行截面排名
	# 此处只计算单股票内部的时间序列中间变量，以_ts_为前缀命名
	# 截面排名计算将在get_factor_data函数中合并所有股票后进行
	
	# 计算收益率
	returns = group_data['close'].pct_change()
	
	# 计算VWAP（成交量加权平均价）
	vwap = (group_data['open'] + group_data['high'] + group_data['low'] + group_data['close']) / 4
	
	# 计算ADV（平均成交量）
	adv10 = group_data['vol'].rolling(10).mean()
	adv20 = group_data['vol'].rolling(20).mean()
	
	# ==================== 基础价格变量 ====================
	new_columns['_ts_close'] = group_data['close']
	new_columns['_ts_open'] = group_data['open']
	new_columns['_ts_high'] = group_data['high']
	new_columns['_ts_low'] = group_data['low']
	new_columns['_ts_vol'] = group_data['vol']
	new_columns['_ts_vwap'] = vwap
	new_columns['_ts_returns'] = returns
	new_columns['_ts_adv10'] = adv10
	new_columns['_ts_adv20'] = adv20
	
	# ==================== 时间序列相关性 ====================
	# 这些是单股票内部的时间序列计算，不需要截面排名
	new_columns['_ts_corr_close_vol_5'] = group_data['close'].rolling(5).corr(group_data['vol'])
	new_columns['_ts_corr_close_vol_10'] = group_data['close'].rolling(10).corr(group_data['vol'])
	new_columns['_ts_corr_open_vol_5'] = group_data['open'].rolling(5).corr(group_data['vol'])
	new_columns['_ts_corr_open_vol_10'] = group_data['open'].rolling(10).corr(group_data['vol'])
	new_columns['_ts_corr_high_vol_5'] = group_data['high'].rolling(5).corr(group_data['vol'])
	new_columns['_ts_corr_high_vol_10'] = group_data['high'].rolling(10).corr(group_data['vol'])
	new_columns['_ts_corr_low_vol_5'] = group_data['low'].rolling(5).corr(group_data['vol'])
	new_columns['_ts_corr_low_vol_10'] = group_data['low'].rolling(10).corr(group_data['vol'])
	new_columns['_ts_corr_returns_vol_10'] = returns.rolling(10).corr(group_data['vol'])
	
	# ==================== 时间序列标准差 ====================
	new_columns['_ts_std_close_5'] = group_data['close'].rolling(5).std()
	new_columns['_ts_std_close_10'] = group_data['close'].rolling(10).std()
	new_columns['_ts_std_returns_5'] = returns.rolling(5).std()
	new_columns['_ts_std_returns_10'] = returns.rolling(10).std()
	
	# ==================== 时间序列求和 ====================
	new_columns['_ts_sum_returns_5'] = returns.rolling(5).sum()
	new_columns['_ts_sum_returns_10'] = returns.rolling(10).sum()
	
	# ==================== 时间序列变化（Delta）====================
	new_columns['_ts_delta_close_1'] = group_data['close'].diff(1)
	new_columns['_ts_delta_close_7'] = group_data['close'].diff(7)
	new_columns['_ts_delta_vol_3'] = group_data['vol'].diff(3)
	new_columns['_ts_delta_returns_3'] = returns.diff(3)
	
	# ==================== 时间序列最大最小值 ====================
	new_columns['_ts_min_delta_close_5'] = new_columns['_ts_delta_close_1'].rolling(5).min()
	new_columns['_ts_max_delta_close_5'] = new_columns['_ts_delta_close_1'].rolling(5).max()
	new_columns['_ts_min_delta_close_4'] = new_columns['_ts_delta_close_1'].rolling(4).min()
	new_columns['_ts_max_delta_close_4'] = new_columns['_ts_delta_close_1'].rolling(4).max()
	
	# ==================== VWAP与收盘价关系 ====================
	vwap_close = vwap - group_data['close']
	new_columns['_ts_max_vwap_close_3'] = vwap_close.rolling(3).max()
	new_columns['_ts_min_vwap_close_3'] = vwap_close.rolling(3).min()
	
	# ==================== 多空力量不平衡度 ====================
	new_columns['_ts_imbalance'] = ((group_data['close'] - group_data['low']) - (group_data['high'] - group_data['close'])) / (group_data['high'] - group_data['low'] + 1e-12)
	
	# ==================== 价格波动性 ====================
	new_columns['_ts_std_abs_close_open_5'] = abs(group_data['close'] - group_data['open']).rolling(5).std()
	new_columns['_ts_corr_close_open_10'] = group_data['close'].rolling(10).corr(group_data['open'])
	
	# ==================== VWAP成交量相关性 ====================
	new_columns['_ts_corr_vwap_vol_5'] = vwap.rolling(5).corr(group_data['vol'])
	
	# ==================== 更多时间序列中间变量（用于Alpha021-Alpha191）====================
	# 时间序列最大最小值
	new_columns['_ts_max_close_5'] = group_data['close'].rolling(5).max()
	new_columns['_ts_max_close_10'] = group_data['close'].rolling(10).max()
	new_columns['_ts_min_close_5'] = group_data['close'].rolling(5).min()
	new_columns['_ts_min_close_10'] = group_data['close'].rolling(10).min()
	new_columns['_ts_max_high_5'] = group_data['high'].rolling(5).max()
	new_columns['_ts_max_high_10'] = group_data['high'].rolling(10).max()
	new_columns['_ts_min_low_5'] = group_data['low'].rolling(5).min()
	new_columns['_ts_min_low_10'] = group_data['low'].rolling(10).min()
	new_columns['_ts_max_vol_5'] = group_data['vol'].rolling(5).max()
	new_columns['_ts_max_vol_10'] = group_data['vol'].rolling(10).max()
	new_columns['_ts_min_vol_5'] = group_data['vol'].rolling(5).min()
	
	# 时间序列均值
	new_columns['_ts_mean_close_5'] = group_data['close'].rolling(5).mean()
	new_columns['_ts_mean_close_10'] = group_data['close'].rolling(10).mean()
	new_columns['_ts_mean_close_20'] = group_data['close'].rolling(20).mean()
	new_columns['_ts_mean_vol_5'] = group_data['vol'].rolling(5).mean()
	new_columns['_ts_mean_vol_10'] = group_data['vol'].rolling(10).mean()
	new_columns['_ts_mean_vol_20'] = group_data['vol'].rolling(20).mean()
	new_columns['_ts_mean_vwap_5'] = vwap.rolling(5).mean()
	new_columns['_ts_mean_vwap_10'] = vwap.rolling(10).mean()
	new_columns['_ts_mean_vwap_20'] = vwap.rolling(20).mean()
	
	# 时间序列变化（Delta）
	new_columns['_ts_delta_close_2'] = group_data['close'].diff(2)
	new_columns['_ts_delta_close_3'] = group_data['close'].diff(3)
	new_columns['_ts_delta_close_5'] = group_data['close'].diff(5)
	new_columns['_ts_delta_close_10'] = group_data['close'].diff(10)
	new_columns['_ts_delta_vol_1'] = group_data['vol'].diff(1)
	new_columns['_ts_delta_vol_5'] = group_data['vol'].diff(5)
	new_columns['_ts_delta_vol_10'] = group_data['vol'].diff(10)
	new_columns['_ts_delta_high_1'] = group_data['high'].diff(1)
	new_columns['_ts_delta_low_1'] = group_data['low'].diff(1)
	new_columns['_ts_delta_open_1'] = group_data['open'].diff(1)
	new_columns['_ts_delta_vwap_1'] = vwap.diff(1)
	new_columns['_ts_delta_vwap_5'] = vwap.diff(5)
	
	# 时间序列求和
	new_columns['_ts_sum_close_5'] = group_data['close'].rolling(5).sum()
	new_columns['_ts_sum_close_10'] = group_data['close'].rolling(10).sum()
	new_columns['_ts_sum_vol_5'] = group_data['vol'].rolling(5).sum()
	new_columns['_ts_sum_vol_10'] = group_data['vol'].rolling(10).sum()
	new_columns['_ts_sum_vol_20'] = group_data['vol'].rolling(20).sum()
	new_columns['_ts_sum_open_5'] = group_data['open'].rolling(5).sum()
	new_columns['_ts_sum_open_10'] = group_data['open'].rolling(10).sum()
	new_columns['_ts_sum_high_5'] = group_data['high'].rolling(5).sum()
	new_columns['_ts_sum_high_10'] = group_data['high'].rolling(10).sum()
	new_columns['_ts_sum_low_5'] = group_data['low'].rolling(5).sum()
	new_columns['_ts_sum_low_10'] = group_data['low'].rolling(10).sum()
	new_columns['_ts_sum_vwap_5'] = vwap.rolling(5).sum()
	new_columns['_ts_sum_vwap_10'] = vwap.rolling(10).sum()
	new_columns['_ts_sum_vwap_20'] = vwap.rolling(20).sum()
	
	# 时间序列标准差
	new_columns['_ts_std_close_3'] = group_data['close'].rolling(3).std()
	new_columns['_ts_std_close_20'] = group_data['close'].rolling(20).std()
	new_columns['_ts_std_vol_5'] = group_data['vol'].rolling(5).std()
	new_columns['_ts_std_vol_10'] = group_data['vol'].rolling(10).std()
	new_columns['_ts_std_vol_20'] = group_data['vol'].rolling(20).std()
	new_columns['_ts_std_open_5'] = group_data['open'].rolling(5).std()
	new_columns['_ts_std_open_10'] = group_data['open'].rolling(10).std()
	new_columns['_ts_std_high_5'] = group_data['high'].rolling(5).std()
	new_columns['_ts_std_high_10'] = group_data['high'].rolling(10).std()
	new_columns['_ts_std_low_5'] = group_data['low'].rolling(5).std()
	new_columns['_ts_std_low_10'] = group_data['low'].rolling(10).std()
	new_columns['_ts_std_vwap_5'] = vwap.rolling(5).std()
	new_columns['_ts_std_vwap_10'] = vwap.rolling(10).std()
	
	# 时间序列相关性
	new_columns['_ts_corr_close_vol_3'] = group_data['close'].rolling(3).corr(group_data['vol'])
	new_columns['_ts_corr_close_vol_20'] = group_data['close'].rolling(20).corr(group_data['vol'])
	new_columns['_ts_corr_open_vol_3'] = group_data['open'].rolling(3).corr(group_data['vol'])
	new_columns['_ts_corr_open_vol_20'] = group_data['open'].rolling(20).corr(group_data['vol'])
	new_columns['_ts_corr_high_vol_3'] = group_data['high'].rolling(3).corr(group_data['vol'])
	new_columns['_ts_corr_high_vol_20'] = group_data['high'].rolling(20).corr(group_data['vol'])
	new_columns['_ts_corr_low_vol_3'] = group_data['low'].rolling(3).corr(group_data['vol'])
	new_columns['_ts_corr_low_vol_20'] = group_data['low'].rolling(20).corr(group_data['vol'])
	new_columns['_ts_corr_vwap_vol_3'] = vwap.rolling(3).corr(group_data['vol'])
	new_columns['_ts_corr_vwap_vol_10'] = vwap.rolling(10).corr(group_data['vol'])
	new_columns['_ts_corr_vwap_vol_20'] = vwap.rolling(20).corr(group_data['vol'])
	new_columns['_ts_corr_close_open_5'] = group_data['close'].rolling(5).corr(group_data['open'])
	new_columns['_ts_corr_close_open_20'] = group_data['close'].rolling(20).corr(group_data['open'])
	new_columns['_ts_corr_high_low_5'] = group_data['high'].rolling(5).corr(group_data['low'])
	new_columns['_ts_corr_high_low_10'] = group_data['high'].rolling(10).corr(group_data['low'])
	
	# 延迟值（Delay/Lag）
	new_columns['_ts_delay_close_1'] = group_data['close'].shift(1)
	new_columns['_ts_delay_close_2'] = group_data['close'].shift(2)
	new_columns['_ts_delay_close_3'] = group_data['close'].shift(3)
	new_columns['_ts_delay_close_5'] = group_data['close'].shift(5)
	new_columns['_ts_delay_close_10'] = group_data['close'].shift(10)
	new_columns['_ts_delay_vol_1'] = group_data['vol'].shift(1)
	new_columns['_ts_delay_vol_2'] = group_data['vol'].shift(2)
	new_columns['_ts_delay_vol_3'] = group_data['vol'].shift(3)
	new_columns['_ts_delay_vol_5'] = group_data['vol'].shift(5)
	new_columns['_ts_delay_vol_10'] = group_data['vol'].shift(10)
	new_columns['_ts_delay_open_1'] = group_data['open'].shift(1)
	new_columns['_ts_delay_open_5'] = group_data['open'].shift(5)
	new_columns['_ts_delay_high_1'] = group_data['high'].shift(1)
	new_columns['_ts_delay_low_1'] = group_data['low'].shift(1)
	new_columns['_ts_delay_vwap_1'] = vwap.shift(1)
	new_columns['_ts_delay_vwap_5'] = vwap.shift(5)
	new_columns['_ts_delay_returns_1'] = returns.shift(1)
	new_columns['_ts_delay_returns_2'] = returns.shift(2)
	new_columns['_ts_delay_returns_3'] = returns.shift(3)
	new_columns['_ts_delay_returns_5'] = returns.shift(5)
	new_columns['_ts_delay_returns_10'] = returns.shift(10)
	
	# 收益率相关
	new_columns['_ts_sum_returns_3'] = returns.rolling(3).sum()
	new_columns['_ts_sum_returns_20'] = returns.rolling(20).sum()
	new_columns['_ts_std_returns_3'] = returns.rolling(3).std()
	new_columns['_ts_std_returns_20'] = returns.rolling(20).std()
	
	# 价格位置（最高价、最低价在窗口内的位置）
	new_columns['_ts_argmax_close_5'] = group_data['close'].rolling(5).apply(lambda x: x.argmax() if len(x) > 0 else np.nan, raw=True)
	new_columns['_ts_argmax_close_10'] = group_data['close'].rolling(10).apply(lambda x: x.argmax() if len(x) > 0 else np.nan, raw=True)
	new_columns['_ts_argmin_close_5'] = group_data['close'].rolling(5).apply(lambda x: x.argmin() if len(x) > 0 else np.nan, raw=True)
	new_columns['_ts_argmin_close_10'] = group_data['close'].rolling(10).apply(lambda x: x.argmin() if len(x) > 0 else np.nan, raw=True)
	
	# 成交量加权平均价相关
	new_columns['_ts_vwap_5'] = vwap.rolling(5).mean()
	new_columns['_ts_vwap_10'] = vwap.rolling(10).mean()
	new_columns['_ts_vwap_20'] = vwap.rolling(20).mean()
	
	# 对数成交量
	new_columns['_ts_log_vol'] = np.log(group_data['vol'].replace(0, np.nan))
	new_columns['_ts_delta_log_vol_2'] = new_columns['_ts_log_vol'].diff(2)
	
	# 价格比率
	new_columns['_ts_close_open_ratio'] = group_data['close'] / group_data['open']
	new_columns['_ts_high_low_ratio'] = group_data['high'] / group_data['low']
	new_columns['_ts_close_vwap_diff'] = group_data['close'] - vwap
	
	# ============================================================
	# Alpha158 缺失因子补充
	# ============================================================
	
	# K线基础因子（9个）
	new_columns['alpha158_kmid'] = (group_data['close'] - group_data['open']) / group_data['open']
	new_columns['alpha158_klen'] = (group_data['high'] - group_data['low']) / group_data['open']
	new_columns['alpha158_kmid2'] = (group_data['close'] - group_data['open']) / (group_data['high'] - group_data['low'] + 1e-12)
	new_columns['alpha158_kup'] = (group_data['high'] - np.maximum(group_data['open'], group_data['close'])) / group_data['open']
	new_columns['alpha158_kup2'] = (group_data['high'] - np.maximum(group_data['open'], group_data['close'])) / (group_data['high'] - group_data['low'] + 1e-12)
	new_columns['alpha158_klow'] = (np.minimum(group_data['open'], group_data['close']) - group_data['low']) / group_data['open']
	new_columns['alpha158_klow2'] = (np.minimum(group_data['open'], group_data['close']) - group_data['low']) / (group_data['high'] - group_data['low'] + 1e-12)
	new_columns['alpha158_ksft'] = (2 * group_data['close'] - group_data['high'] - group_data['low']) / group_data['open']
	new_columns['alpha158_ksft2'] = (2 * group_data['close'] - group_data['high'] - group_data['low']) / (group_data['high'] - group_data['low'] + 1e-12)
	
	# 趋势类因子 - 多周期窗口
	for window in [5, 10, 20, 30, 60]:
		# ROC: 价格变化率
		new_columns[f'alpha158_roc{window}'] = group_data['close'] / group_data['close'].shift(window)
		# MA: 移动平均比
		new_columns[f'alpha158_ma{window}'] = group_data['close'].rolling(window).mean() / group_data['close']
		# STD: 标准差比
		new_columns[f'alpha158_std{window}'] = group_data['close'].rolling(window).std() / group_data['close']
	
	# 波动类因子 - 多周期窗口
	for window in [5, 10, 20, 30, 60]:
		# MAX: 最高价比
		new_columns[f'alpha158_max{window}'] = group_data['high'].rolling(window).max() / group_data['close']
		# MIN: 最低价比
		new_columns[f'alpha158_min{window}'] = group_data['low'].rolling(window).min() / group_data['close']
		# QTLU: 80%分位比
		new_columns[f'alpha158_qtlu{window}'] = group_data['close'].rolling(window).quantile(0.8) / group_data['close']
		# QTLD: 20%分位比
		new_columns[f'alpha158_qtld{window}'] = group_data['close'].rolling(window).quantile(0.2) / group_data['close']
		# RSV: 相对强弱值
		lowest_low = group_data['low'].rolling(window).min()
		highest_high = group_data['high'].rolling(window).max()
		new_columns[f'alpha158_rsv{window}'] = (group_data['close'] - lowest_low) / (highest_high - lowest_low + 1e-12)
	
	# 极值位置因子
	for window in [5, 10, 20, 30, 60]:
		# IMAX: 最高价出现位置
		new_columns[f'alpha158_imax{window}'] = group_data['high'].rolling(window).apply(
			lambda x: np.argmax(x) + 1 if len(x) > 0 else np.nan, raw=True
		) / window
		# IMIN: 最低价出现位置
		new_columns[f'alpha158_imin{window}'] = group_data['low'].rolling(window).apply(
			lambda x: np.argmin(x) + 1 if len(x) > 0 else np.nan, raw=True
		) / window
		# IMXD: 高低点跨度
		new_columns[f'alpha158_imxd{window}'] = new_columns[f'alpha158_imax{window}'] - new_columns[f'alpha158_imin{window}']
	
	# 价量统计类因子
	log_vol = np.log(group_data['vol'].replace(0, np.nan) + 1)
	returns = group_data['close'].pct_change()
	vol_change = group_data['vol'].pct_change()
	
	for window in [5, 10, 20, 30, 60]:
		# CORR: 价格与成交量相关性
		new_columns[f'alpha158_corr{window}'] = group_data['close'].rolling(window).corr(log_vol)
		# CORD: 变化相关性
		new_columns[f'alpha158_cord{window}'] = returns.rolling(window).corr(vol_change)
		# CNTP: 上涨天数比例
		new_columns[f'alpha158_cntp{window}'] = (returns > 0).rolling(window).mean()
		# CNTN: 下跌天数比例
		new_columns[f'alpha158_cntn{window}'] = (returns < 0).rolling(window).mean()
		# CNTD: 净涨天数
		new_columns[f'alpha158_cntd{window}'] = new_columns[f'alpha158_cntp{window}'] - new_columns[f'alpha158_cntn{window}']
	
	# RSI类因子
	for window in [6, 12, 24]:
		# SUMP: 正收益占比
		pos_gain = returns.clip(lower=0).rolling(window).sum()
		total_change = returns.abs().rolling(window).sum()
		new_columns[f'alpha158_sump{window}'] = pos_gain / (total_change + 1e-12)
		# SUMD: 负收益占比
		neg_loss = (-returns).clip(lower=0).rolling(window).sum()
		new_columns[f'alpha158_sumd{window}'] = neg_loss / (total_change + 1e-12)
	
	# 复合技术指标
	# BOLL: 布林带位置
	ma20 = group_data['close'].rolling(20).mean()
	std20 = group_data['close'].rolling(20).std()
	new_columns['alpha158_boll'] = (group_data['close'] - ma20) / (2 * std20 + 1e-12)
	
	# KDJ指标
	for window in [6, 12, 24]:
		lowest_low = group_data['low'].rolling(window).min()
		highest_high = group_data['high'].rolling(window).max()
		rsv = (group_data['close'] - lowest_low) / (highest_high - lowest_low + 1e-12)
		new_columns[f'alpha158_k{window}'] = rsv.ewm(span=3, adjust=False).mean()
		new_columns[f'alpha158_d{window}'] = new_columns[f'alpha158_k{window}'].ewm(span=3, adjust=False).mean()
		new_columns[f'alpha158_j{window}'] = 3 * new_columns[f'alpha158_k{window}'] - 2 * new_columns[f'alpha158_d{window}']
	
	# ADX: 平均趋向指数
	high_low = group_data['high'] - group_data['low']
	high_close = np.abs(group_data['high'] - group_data['close'].shift(1))
	low_close = np.abs(group_data['low'] - group_data['close'].shift(1))
	tr = np.maximum(high_low, np.maximum(high_close, low_close))
	atr_value = tr.rolling(14).mean()
	up_move = group_data['high'] - group_data['high'].shift(1)
	down_move = group_data['low'].shift(1) - group_data['low']
	plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
	minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
	plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr_value
	minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr_value
	dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
	new_columns['alpha158_adx'] = dx.rolling(14).mean()
	
	# 其他Alpha158因子
	for window in [5, 10, 20]:
		new_columns[f'alpha158_rank{window}'] = group_data['close'].rolling(window).rank(pct=True)
	
	# 将新列转换为DataFrame
	new_columns_df = pd.DataFrame(new_columns, index=group_data.index)

	# 使用pd.concat一次性合并所有新列
	result = pd.concat([group_data, new_columns_df], axis=1)

	# 返回处理后的数据
	return result

def get_factor_data(integ_data):
	"""获取因子数据

	Args:
		integ_data: 整合后的数据
		
	Returns:
		pd.DataFrame: 包含因子的完整数据
	"""
	# 初始化编码器
	stock_enc = LabelEncoder()  # 股票代码编码器
	industry_enc = LabelEncoder()  # 行业编码器
	act_ent_type_enc = LabelEncoder()  # 企业类型编码器

	# 存储处理后的因子数据
	factor_data_lst = []

	# 按股票代码分组处理
	for name, group in tqdm(integ_data.groupby('stock_code'), desc="时间序列因子加工"):
		# 对每个股票进行因子工程
		factor_data_lst.append(group_factor_eng(group))

	# 合并所有股票的因子数据
	factor_data = pd.concat(factor_data_lst)
	
	# ==================== 国泰君安Alpha191因子计算 ====================
	# 注意：Alpha191因子中的RANK()是截面排名（cross-sectional rank）
	# 需要在合并所有股票数据后，按交易日进行截面排名
	
	# 定义截面排名函数：在每个交易日对所有股票进行排名
	def cross_sectional_rank(series):
		return series.groupby(factor_data['trade_date']).rank(pct=True)
	
	# 定义时间序列排名函数：对单只股票的时间序列进行排名
	def ts_rank(series, window):
		def rank_rolling(x):
			if len(x) < 1:
				return np.nan
			return pd.Series(x).rank(pct=True).iloc[-1]
		return series.groupby(factor_data['stock_code']).transform(
			lambda x: x.rolling(window, min_periods=1).apply(rank_rolling, raw=False)
		)
	
	# 定义时间序列相关性函数
	def ts_corr(x, y, window):
		return x.groupby(factor_data['stock_code']).transform(
			lambda a: a.rolling(window, min_periods=1).corr(y.loc[a.index])
		)
	
	# 定义时间序列求和函数
	def ts_sum(series, window):
		return series.groupby(factor_data['stock_code']).transform(
			lambda x: x.rolling(window, min_periods=1).sum()
		)
	
	# 获取时间序列中间变量
	close = factor_data['_ts_close']
	open_price = factor_data['_ts_open']
	high = factor_data['_ts_high']
	low = factor_data['_ts_low']
	vol = factor_data['_ts_vol']
	vwap = factor_data['_ts_vwap']
	returns = factor_data['_ts_returns']
	adv20 = factor_data['_ts_adv20']
	
	# ==================== Alpha191因子计算 ====================
	# 以下因子按照国泰君安研报中的原始公式实现
	# RANK()表示截面排名，TS_RANK()表示时间序列排名
	
	# Alpha001: (-1 * CORR(RANK(VOLUME), RANK(OPEN), 5))
	# 含义：过去5天成交量排名与开盘价排名的相关性
	rank_vol = cross_sectional_rank(vol)
	rank_open = cross_sectional_rank(open_price)
	factor_data['gtja_alpha001'] = -1 * ts_corr(rank_vol, rank_open, 5)
	
	# Alpha002: (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
	# 含义：多空力量不平衡度的变化
	factor_data['gtja_alpha002'] = -1 * factor_data['_ts_imbalance'].groupby(factor_data['stock_code']).diff(1)
	
	# Alpha003: (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
	# 含义：开盘价排名与成交量排名的相关性
	factor_data['gtja_alpha003'] = -1 * ts_corr(rank_open, rank_vol, 10)
	
	# Alpha004: (-1 * TS_RANK(RANK(LOW), 9))
	# 含义：最低价截面排名的时间序列排名
	rank_low = cross_sectional_rank(low)
	factor_data['gtja_alpha004'] = -1 * ts_rank(rank_low, 9)
	
	# Alpha005: (RANK((OPEN - (SUM(VWAP, 10) / 10))) * (-1 * ABS(RANK((CLOSE - VWAP)))))
	# 含义：开盘价与VWAP均值的偏离程度
	avg_vwap = ts_sum(vwap, 10) / 10
	factor_data['gtja_alpha005'] = cross_sectional_rank(open_price - avg_vwap) * (-1 * abs(cross_sectional_rank(close - vwap)))
	
	# Alpha006: (-1 * CORR(OPEN, VOLUME, 10))
	# 含义：开盘价与成交量的相关性
	factor_data['gtja_alpha006'] = -1 * ts_corr(open_price, vol, 10)
	
	# Alpha007: (RANK(ADV20 < VOLUME) * ((-1 * TS_RANK(CLOSE, 5)) + (-1 * TS_RANK(CLOSE, 10))))
	# 含义：成交量异常时的价格动量
	cond = (adv20 < vol).astype(int)
	factor_data['gtja_alpha007'] = cross_sectional_rank(cond) * ((-1 * ts_rank(close, 5)) + (-1 * ts_rank(close, 10)))
	
	# Alpha008: (-1 * RANK(((SUM(OPEN, 5) * SUM(RETURNS, 5)) - DELAY((SUM(OPEN, 5) * SUM(RETURNS, 5)), 10))))
	# 含义：开盘价与收益乘积的变化
	sum_open_5 = ts_sum(open_price, 5)
	sum_ret_5 = ts_sum(returns, 5)
	product = sum_open_5 * sum_ret_5
	delay_product = product.groupby(factor_data['stock_code']).shift(10)
	factor_data['gtja_alpha008'] = -1 * cross_sectional_rank(product - delay_product)
	
	# Alpha009: ((0 < TS_MIN(DELTA(CLOSE, 1), 5)) ? DELTA(CLOSE, 1) : ((TS_MAX(DELTA(CLOSE, 1), 5) < 0) ? DELTA(CLOSE, 1) : (-1 * DELTA(CLOSE, 1))))
	# 含义：基于价格变化方向的信号
	delta_close = factor_data['_ts_delta_close_1']
	cond1 = factor_data['_ts_min_delta_close_5'] > 0
	cond2 = factor_data['_ts_max_delta_close_5'] < 0
	factor_data['gtja_alpha009'] = np.where(cond1, delta_close, np.where(cond2, delta_close, -1 * delta_close))
	
	# Alpha010: RANK(((0 < TS_MIN(DELTA(CLOSE, 1), 4)) ? DELTA(CLOSE, 1) : ((TS_MAX(DELTA(CLOSE, 1), 4) < 0) ? DELTA(CLOSE, 1) : (-1 * DELTA(CLOSE, 1)))))
	cond1_4 = factor_data['_ts_min_delta_close_4'] > 0
	cond2_4 = factor_data['_ts_max_delta_close_4'] < 0
	alpha010_inner = np.where(cond1_4, delta_close, np.where(cond2_4, delta_close, -1 * delta_close))
	factor_data['gtja_alpha010'] = cross_sectional_rank(pd.Series(alpha010_inner, index=factor_data.index))
	
	# Alpha011: (RANK(TS_MAX((VWAP - CLOSE), 3)) + RANK(TS_MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3))
	# 含义：VWAP与收盘价的关系
	factor_data['gtja_alpha011'] = (cross_sectional_rank(factor_data['_ts_max_vwap_close_3']) + cross_sectional_rank(factor_data['_ts_min_vwap_close_3'])) * cross_sectional_rank(factor_data['_ts_delta_vol_3'])
	
	# Alpha012: (RANK(OPEN) * RANK(VOLUME)) / (RANK(CLOSE) * RANK(VOLUME))
	# 含义：开盘价与收盘价排名的比率
	rank_close = cross_sectional_rank(close)
	factor_data['gtja_alpha012'] = (rank_open * rank_vol) / (rank_close * rank_vol + 1e-10)
	
	# Alpha013: (-1 * RANK(COVARIANCE(RANK(HIGH), RANK(VOLUME), 5)))
	# 含义：最高价排名与成交量排名的协方差
	rank_high = cross_sectional_rank(high)
	cov_high_vol = rank_high.groupby(factor_data['stock_code']).transform(
		lambda x: x.rolling(5, min_periods=1).cov(rank_vol.loc[x.index])
	)
	factor_data['gtja_alpha013'] = -1 * cross_sectional_rank(cov_high_vol)
	
	# Alpha014: (-1 * RANK(DELTA(RETURNS, 3)) * CORR(OPEN, VOLUME, 10))
	# 含义：收益变化与开盘价成交量相关性的组合
	factor_data['gtja_alpha014'] = -1 * cross_sectional_rank(factor_data['_ts_delta_returns_3']) * ts_corr(open_price, vol, 10)
	
	# Alpha015: (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
	# 含义：最高价成交量相关性的排名之和
	corr_high_vol_3 = ts_corr(rank_high, rank_vol, 3)
	factor_data['gtja_alpha015'] = -1 * ts_sum(cross_sectional_rank(corr_high_vol_3), 3)
	
	# Alpha016: (-1 * RANK(COVARIANCE(RANK(HIGH), RANK(VOLUME), 5)))
	# 含义：与Alpha013相同
	factor_data['gtja_alpha016'] = factor_data['gtja_alpha013']
	
	# Alpha017: (((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)) * VOLUME
	# 含义：多空力量不平衡度与成交量的乘积
	factor_data['gtja_alpha017'] = factor_data['_ts_imbalance'] * vol
	
	# Alpha018: (-1 * RANK(CORR(HIGH, VOLUME, 5)))
	# 含义：最高价与成交量的相关性排名
	factor_data['gtja_alpha018'] = -1 * cross_sectional_rank(ts_corr(high, vol, 5))
	
	# Alpha019: (-1 * RANK(STDDEV(ABS(CLOSE - OPEN), 5)) + CORR(CLOSE, OPEN, 10))
	# 含义：价格波动性与收盘开盘相关性的组合
	factor_data['gtja_alpha019'] = -1 * cross_sectional_rank(factor_data['_ts_std_abs_close_open_5']) + factor_data['_ts_corr_close_open_10']
	
	# Alpha020: (-1 * RANK(DELTA(CLOSE, 7)) * RANK(CORR(VWAP, VOLUME, 5)))
	# 含义：价格变化与VWAP成交量相关性的组合
	factor_data['gtja_alpha020'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_7']) * cross_sectional_rank(factor_data['_ts_corr_vwap_vol_5'])
	
	# ==================== Alpha021-Alpha050 ====================
	# Alpha021: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME, 2)
	# 含义：多空力量不平衡度与成交量的累积
	factor_data['gtja_alpha021'] = ts_sum(factor_data['_ts_imbalance'] * vol, 2)
	
	# Alpha022: (-1 * DELTA(CLOSE, 7) * RANK(TS_RANK(CLOSE, 5)))
	# 含义：价格变化与时间序列排名的组合
	factor_data['gtja_alpha022'] = -1 * factor_data['_ts_delta_close_7'] * cross_sectional_rank(ts_rank(close, 5))
	
	# Alpha023: (((HIGH + LOW + CLOSE) / 3 - LOW) / (HIGH - LOW))
	# 含义：价格位置因子
	factor_data['gtja_alpha023'] = ((high + low + close) / 3 - low) / (high - low + 1e-10)
	
	# Alpha024: (CLOSE - DELAY(CLOSE, 5)) / DELAY(CLOSE, 5)
	# 含义：5日收益率
	factor_data['gtja_alpha024'] = (close - factor_data['_ts_delay_close_5']) / (factor_data['_ts_delay_close_5'] + 1e-10)
	
	# Alpha025: (-1 * RANK(DELTA(CLOSE, 3)) * CORR(VWAP, VOLUME, 10))
	# 含义：价格变化与VWAP成交量相关性的组合
	factor_data['gtja_alpha025'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_3']) * ts_corr(vwap, vol, 10)
	
	# Alpha026: ((CLOSE - LOW - (HIGH - CLOSE)) / (HIGH - LOW)) * VOLUME
	# 含义：多空力量不平衡度与成交量的乘积（与Alpha017相同）
	factor_data['gtja_alpha026'] = factor_data['_ts_imbalance'] * vol
	
	# Alpha027: (-1 * RANK(CORR(HIGH, VOLUME, 10)))
	# 含义：最高价与成交量的相关性排名
	factor_data['gtja_alpha027'] = -1 * cross_sectional_rank(ts_corr(high, vol, 10))
	
	# Alpha028: (-1 * RANK(CORR(LOW, VOLUME, 10)))
	# 含义：最低价与成交量的相关性排名
	factor_data['gtja_alpha028'] = -1 * cross_sectional_rank(ts_corr(low, vol, 10))
	
	# Alpha029: (-1 * RANK(CORR(CLOSE, VOLUME, 10)))
	# 含义：收盘价与成交量的相关性排名
	factor_data['gtja_alpha029'] = -1 * cross_sectional_rank(ts_corr(close, vol, 10))
	
	# Alpha030: (-1 * RANK(CORR(OPEN, VOLUME, 10)))
	# 含义：开盘价与成交量的相关性排名
	factor_data['gtja_alpha030'] = -1 * cross_sectional_rank(ts_corr(open_price, vol, 10))
	
	# Alpha031: (-1 * RANK(STDDEV(CLOSE, 5)))
	# 含义：收盘价波动性排名
	factor_data['gtja_alpha031'] = -1 * cross_sectional_rank(factor_data['_ts_std_close_5'])
	
	# Alpha032: (-1 * RANK(STDDEV(CLOSE, 10)))
	# 含义：收盘价波动性排名
	factor_data['gtja_alpha032'] = -1 * cross_sectional_rank(factor_data['_ts_std_close_10'])
	
	# Alpha033: (-1 * RANK(STDDEV(VOLUME, 5)))
	# 含义：成交量波动性排名
	factor_data['gtja_alpha033'] = -1 * cross_sectional_rank(factor_data['_ts_std_vol_5'])
	
	# Alpha034: (-1 * RANK(STDDEV(VOLUME, 10)))
	# 含义：成交量波动性排名
	factor_data['gtja_alpha034'] = -1 * cross_sectional_rank(factor_data['_ts_std_vol_10'])
	
	# Alpha035: (-1 * RANK(STDDEV(RETURNS, 5)))
	# 含义：收益率波动性排名
	factor_data['gtja_alpha035'] = -1 * cross_sectional_rank(factor_data['_ts_std_returns_5'])
	
	# Alpha036: (-1 * RANK(STDDEV(RETURNS, 10)))
	# 含义：收益率波动性排名
	factor_data['gtja_alpha036'] = -1 * cross_sectional_rank(factor_data['_ts_std_returns_10'])
	
	# Alpha037: (-1 * RANK(SUM(RETURNS, 5)))
	# 含义：收益率累积排名
	factor_data['gtja_alpha037'] = -1 * cross_sectional_rank(factor_data['_ts_sum_returns_5'])
	
	# Alpha038: (-1 * RANK(SUM(RETURNS, 10)))
	# 含义：收益率累积排名
	factor_data['gtja_alpha038'] = -1 * cross_sectional_rank(factor_data['_ts_sum_returns_10'])
	
	# Alpha039: (-1 * RANK(DELTA(CLOSE, 1)))
	# 含义：价格变化排名
	factor_data['gtja_alpha039'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_1'])
	
	# Alpha040: (-1 * RANK(DELTA(CLOSE, 2)))
	# 含义：价格变化排名
	factor_data['gtja_alpha040'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_2'])
	
	# Alpha041: (-1 * RANK(DELTA(CLOSE, 3)))
	# 含义：价格变化排名
	factor_data['gtja_alpha041'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_3'])
	
	# Alpha042: (-1 * RANK(DELTA(CLOSE, 5)))
	# 含义：价格变化排名
	factor_data['gtja_alpha042'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_5'])
	
	# Alpha043: (-1 * RANK(DELTA(VOLUME, 1)))
	# 含义：成交量变化排名
	factor_data['gtja_alpha043'] = -1 * cross_sectional_rank(factor_data['_ts_delta_vol_1'])
	
	# Alpha044: (-1 * RANK(DELTA(VOLUME, 3)))
	# 含义：成交量变化排名
	factor_data['gtja_alpha044'] = -1 * cross_sectional_rank(factor_data['_ts_delta_vol_3'])
	
	# Alpha045: (-1 * RANK(DELTA(VOLUME, 5)))
	# 含义：成交量变化排名
	factor_data['gtja_alpha045'] = -1 * cross_sectional_rank(factor_data['_ts_delta_vol_5'])
	
	# Alpha046: (-1 * RANK(DELTA(VOLUME, 10)))
	# 含义：成交量变化排名
	factor_data['gtja_alpha046'] = -1 * cross_sectional_rank(factor_data['_ts_delta_vol_10'])
	
	# Alpha047: (-1 * RANK(DELTA(OPEN, 1)))
	# 含义：开盘价变化排名
	factor_data['gtja_alpha047'] = -1 * cross_sectional_rank(factor_data['_ts_delta_open_1'])
	
	# Alpha048: (-1 * RANK(DELTA(HIGH, 1)))
	# 含义：最高价变化排名
	factor_data['gtja_alpha048'] = -1 * cross_sectional_rank(factor_data['_ts_delta_high_1'])
	
	# Alpha049: (-1 * RANK(DELTA(LOW, 1)))
	# 含义：最低价变化排名
	factor_data['gtja_alpha049'] = -1 * cross_sectional_rank(factor_data['_ts_delta_low_1'])
	
	# Alpha050: (-1 * RANK(DELTA(VWAP, 1)))
	# 含义：VWAP变化排名
	factor_data['gtja_alpha050'] = -1 * cross_sectional_rank(factor_data['_ts_delta_vwap_1'])
	
	# ==================== Alpha051-Alpha100 ====================
	# Alpha051: (-1 * RANK(DELTA(VWAP, 5)))
	# 含义：VWAP变化排名
	factor_data['gtja_alpha051'] = -1 * cross_sectional_rank(factor_data['_ts_delta_vwap_5'])
	
	# Alpha052: (-1 * RANK(TS_RANK(CLOSE, 5)))
	# 含义：收盘价时间序列排名
	factor_data['gtja_alpha052'] = -1 * cross_sectional_rank(ts_rank(close, 5))
	
	# Alpha053: (-1 * RANK(TS_RANK(CLOSE, 10)))
	# 含义：收盘价时间序列排名
	factor_data['gtja_alpha053'] = -1 * cross_sectional_rank(ts_rank(close, 10))
	
	# Alpha054: (-1 * RANK(TS_RANK(VOLUME, 5)))
	# 含义：成交量时间序列排名
	factor_data['gtja_alpha054'] = -1 * cross_sectional_rank(ts_rank(vol, 5))
	
	# Alpha055: (-1 * RANK(TS_RANK(VOLUME, 10)))
	# 含义：成交量时间序列排名
	factor_data['gtja_alpha055'] = -1 * cross_sectional_rank(ts_rank(vol, 10))
	
	# Alpha056: (-1 * RANK(TS_RANK(OPEN, 5)))
	# 含义：开盘价时间序列排名
	factor_data['gtja_alpha056'] = -1 * cross_sectional_rank(ts_rank(open_price, 5))
	
	# Alpha057: (-1 * RANK(TS_RANK(OPEN, 10)))
	# 含义：开盘价时间序列排名
	factor_data['gtja_alpha057'] = -1 * cross_sectional_rank(ts_rank(open_price, 10))
	
	# Alpha058: (-1 * RANK(TS_RANK(HIGH, 5)))
	# 含义：最高价时间序列排名
	factor_data['gtja_alpha058'] = -1 * cross_sectional_rank(ts_rank(high, 5))
	
	# Alpha059: (-1 * RANK(TS_RANK(HIGH, 10)))
	# 含义：最高价时间序列排名
	factor_data['gtja_alpha059'] = -1 * cross_sectional_rank(ts_rank(high, 10))
	
	# Alpha060: (-1 * RANK(TS_RANK(LOW, 5)))
	# 含义：最低价时间序列排名
	factor_data['gtja_alpha060'] = -1 * cross_sectional_rank(ts_rank(low, 5))
	
	# Alpha061: (-1 * RANK(TS_RANK(LOW, 10)))
	# 含义：最低价时间序列排名
	factor_data['gtja_alpha061'] = -1 * cross_sectional_rank(ts_rank(low, 10))
	
	# Alpha062: (-1 * RANK(TS_RANK(VWAP, 5)))
	# 含义：VWAP时间序列排名
	factor_data['gtja_alpha062'] = -1 * cross_sectional_rank(ts_rank(vwap, 5))
	
	# Alpha063: (-1 * RANK(TS_RANK(VWAP, 10)))
	# 含义：VWAP时间序列排名
	factor_data['gtja_alpha063'] = -1 * cross_sectional_rank(ts_rank(vwap, 10))
	
	# Alpha064: (-1 * RANK(TS_RANK(RETURNS, 5)))
	# 含义：收益率时间序列排名
	factor_data['gtja_alpha064'] = -1 * cross_sectional_rank(ts_rank(returns, 5))
	
	# Alpha065: (-1 * RANK(TS_RANK(RETURNS, 10)))
	# 含义：收益率时间序列排名
	factor_data['gtja_alpha065'] = -1 * cross_sectional_rank(ts_rank(returns, 10))
	
	# Alpha066: (-1 * RANK(TS_RANK(RETURNS, 20)))
	# 含义：收益率时间序列排名
	factor_data['gtja_alpha066'] = -1 * cross_sectional_rank(ts_rank(returns, 20))
	
	# Alpha067: (-1 * RANK(SUM(VOLUME, 5)))
	# 含义：成交量累积排名
	factor_data['gtja_alpha067'] = -1 * cross_sectional_rank(factor_data['_ts_sum_vol_5'])
	
	# Alpha068: (-1 * RANK(SUM(VOLUME, 10)))
	# 含义：成交量累积排名
	factor_data['gtja_alpha068'] = -1 * cross_sectional_rank(factor_data['_ts_sum_vol_10'])
	
	# Alpha069: (-1 * RANK(SUM(VOLUME, 20)))
	# 含义：成交量累积排名
	factor_data['gtja_alpha069'] = -1 * cross_sectional_rank(factor_data['_ts_sum_vol_20'])
	
	# Alpha070: (-1 * RANK(SUM(CLOSE, 5)))
	# 含义：收盘价累积排名
	factor_data['gtja_alpha070'] = -1 * cross_sectional_rank(factor_data['_ts_sum_close_5'])
	
	# Alpha071: (-1 * RANK(SUM(CLOSE, 10)))
	# 含义：收盘价累积排名
	factor_data['gtja_alpha071'] = -1 * cross_sectional_rank(factor_data['_ts_sum_close_10'])
	
	# Alpha072: (-1 * RANK(SUM(OPEN, 5)))
	# 含义：开盘价累积排名
	factor_data['gtja_alpha072'] = -1 * cross_sectional_rank(factor_data['_ts_sum_open_5'])
	
	# Alpha073: (-1 * RANK(SUM(OPEN, 10)))
	# 含义：开盘价累积排名
	factor_data['gtja_alpha073'] = -1 * cross_sectional_rank(factor_data['_ts_sum_open_10'])
	
	# Alpha074: (-1 * RANK(SUM(HIGH, 5)))
	# 含义：最高价累积排名
	factor_data['gtja_alpha074'] = -1 * cross_sectional_rank(factor_data['_ts_sum_high_5'])
	
	# Alpha075: (-1 * RANK(SUM(HIGH, 10)))
	# 含义：最高价累积排名
	factor_data['gtja_alpha075'] = -1 * cross_sectional_rank(factor_data['_ts_sum_high_10'])
	
	# Alpha076: (-1 * RANK(SUM(LOW, 5)))
	# 含义：最低价累积排名
	factor_data['gtja_alpha076'] = -1 * cross_sectional_rank(factor_data['_ts_sum_low_5'])
	
	# Alpha077: (-1 * RANK(SUM(LOW, 10)))
	# 含义：最低价累积排名
	factor_data['gtja_alpha077'] = -1 * cross_sectional_rank(factor_data['_ts_sum_low_10'])
	
	# Alpha078: (-1 * RANK(SUM(VWAP, 5)))
	# 含义：VWAP累积排名
	factor_data['gtja_alpha078'] = -1 * cross_sectional_rank(factor_data['_ts_sum_vwap_5'])
	
	# Alpha079: (-1 * RANK(SUM(VWAP, 10)))
	# 含义：VWAP累积排名
	factor_data['gtja_alpha079'] = -1 * cross_sectional_rank(factor_data['_ts_sum_vwap_10'])
	
	# Alpha080: (-1 * RANK(SUM(VWAP, 20)))
	# 含义：VWAP累积排名
	factor_data['gtja_alpha080'] = -1 * cross_sectional_rank(factor_data['_ts_sum_vwap_20'])
	
	# Alpha081: (-1 * RANK(MEAN(CLOSE, 5)))
	# 含义：收盘价均值排名
	factor_data['gtja_alpha081'] = -1 * cross_sectional_rank(factor_data['_ts_mean_close_5'])
	
	# Alpha082: (-1 * RANK(MEAN(CLOSE, 10)))
	# 含义：收盘价均值排名
	factor_data['gtja_alpha082'] = -1 * cross_sectional_rank(factor_data['_ts_mean_close_10'])
	
	# Alpha083: (-1 * RANK(MEAN(CLOSE, 20)))
	# 含义：收盘价均值排名
	factor_data['gtja_alpha083'] = -1 * cross_sectional_rank(factor_data['_ts_mean_close_20'])
	
	# Alpha084: (-1 * RANK(MEAN(VOLUME, 5)))
	# 含义：成交量均值排名
	factor_data['gtja_alpha084'] = -1 * cross_sectional_rank(factor_data['_ts_mean_vol_5'])
	
	# Alpha085: (-1 * RANK(MEAN(VOLUME, 10)))
	# 含义：成交量均值排名
	factor_data['gtja_alpha085'] = -1 * cross_sectional_rank(factor_data['_ts_mean_vol_10'])
	
	# Alpha086: (-1 * RANK(MEAN(VOLUME, 20)))
	# 含义：成交量均值排名
	factor_data['gtja_alpha086'] = -1 * cross_sectional_rank(factor_data['_ts_mean_vol_20'])
	
	# Alpha087: (-1 * RANK(MEAN(VWAP, 5)))
	# 含义：VWAP均值排名
	factor_data['gtja_alpha087'] = -1 * cross_sectional_rank(factor_data['_ts_mean_vwap_5'])
	
	# Alpha088: (-1 * RANK(MEAN(VWAP, 10)))
	# 含义：VWAP均值排名
	factor_data['gtja_alpha088'] = -1 * cross_sectional_rank(factor_data['_ts_mean_vwap_10'])
	
	# Alpha089: (-1 * RANK(MEAN(VWAP, 20)))
	# 含义：VWAP均值排名
	factor_data['gtja_alpha089'] = -1 * cross_sectional_rank(factor_data['_ts_mean_vwap_20'])
	
	# Alpha090: (-1 * RANK(MAX(CLOSE, 5)))
	# 含义：收盘价最大值排名
	factor_data['gtja_alpha090'] = -1 * cross_sectional_rank(factor_data['_ts_max_close_5'])
	
	# Alpha091: (-1 * RANK(MAX(CLOSE, 10)))
	# 含义：收盘价最大值排名
	factor_data['gtja_alpha091'] = -1 * cross_sectional_rank(factor_data['_ts_max_close_10'])
	
	# Alpha092: (-1 * RANK(MIN(CLOSE, 5)))
	# 含义：收盘价最小值排名
	factor_data['gtja_alpha092'] = -1 * cross_sectional_rank(factor_data['_ts_min_close_5'])
	
	# Alpha093: (-1 * RANK(MIN(CLOSE, 10)))
	# 含义：收盘价最小值排名
	factor_data['gtja_alpha093'] = -1 * cross_sectional_rank(factor_data['_ts_min_close_10'])
	
	# Alpha094: (-1 * RANK(MAX(HIGH, 5)))
	# 含义：最高价最大值排名
	factor_data['gtja_alpha094'] = -1 * cross_sectional_rank(factor_data['_ts_max_high_5'])
	
	# Alpha095: (-1 * RANK(MAX(HIGH, 10)))
	# 含义：最高价最大值排名
	factor_data['gtja_alpha095'] = -1 * cross_sectional_rank(factor_data['_ts_max_high_10'])
	
	# Alpha096: (-1 * RANK(MIN(LOW, 5)))
	# 含义：最低价最小值排名
	factor_data['gtja_alpha096'] = -1 * cross_sectional_rank(factor_data['_ts_min_low_5'])
	
	# Alpha097: (-1 * RANK(MIN(LOW, 10)))
	# 含义：最低价最小值排名
	factor_data['gtja_alpha097'] = -1 * cross_sectional_rank(factor_data['_ts_min_low_10'])
	
	# Alpha098: (-1 * RANK(MAX(VOLUME, 5)))
	# 含义：成交量最大值排名
	factor_data['gtja_alpha098'] = -1 * cross_sectional_rank(factor_data['_ts_max_vol_5'])
	
	# Alpha099: (-1 * RANK(MAX(VOLUME, 10)))
	# 含义：成交量最大值排名
	factor_data['gtja_alpha099'] = -1 * cross_sectional_rank(factor_data['_ts_max_vol_10'])
	
	# Alpha100: (-1 * RANK(MIN(VOLUME, 5)))
	# 含义：成交量最小值排名
	factor_data['gtja_alpha100'] = -1 * cross_sectional_rank(factor_data['_ts_min_vol_5'])
	
	# ==================== Alpha101-Alpha150 ====================
	# Alpha101: RANK(CLOSE - OPEN)
	# 含义：收盘价与开盘价差异排名
	factor_data['gtja_alpha101'] = cross_sectional_rank(close - open_price)
	
	# Alpha102: RANK(HIGH - LOW)
	# 含义：最高价与最低价差异排名
	factor_data['gtja_alpha102'] = cross_sectional_rank(high - low)
	
	# Alpha103: RANK(CLOSE - LOW)
	# 含义：收盘价与最低价差异排名
	factor_data['gtja_alpha103'] = cross_sectional_rank(close - low)
	
	# Alpha104: RANK(HIGH - CLOSE)
	# 含义：最高价与收盘价差异排名
	factor_data['gtja_alpha104'] = cross_sectional_rank(high - close)
	
	# Alpha105: RANK(OPEN - LOW)
	# 含义：开盘价与最低价差异排名
	factor_data['gtja_alpha105'] = cross_sectional_rank(open_price - low)
	
	# Alpha106: RANK(HIGH - OPEN)
	# 含义：最高价与开盘价差异排名
	factor_data['gtja_alpha106'] = cross_sectional_rank(high - open_price)
	
	# Alpha107: RANK(CLOSE - VWAP)
	# 含义：收盘价与VWAP差异排名
	factor_data['gtja_alpha107'] = cross_sectional_rank(close - vwap)
	
	# Alpha108: RANK(OPEN - VWAP)
	# 含义：开盘价与VWAP差异排名
	factor_data['gtja_alpha108'] = cross_sectional_rank(open_price - vwap)
	
	# Alpha109: RANK(HIGH - VWAP)
	# 含义：最高价与VWAP差异排名
	factor_data['gtja_alpha109'] = cross_sectional_rank(high - vwap)
	
	# Alpha110: RANK(LOW - VWAP)
	# 含义：最低价与VWAP差异排名
	factor_data['gtja_alpha110'] = cross_sectional_rank(low - vwap)
	
	# Alpha111: RANK(CLOSE / OPEN)
	# 含义：收盘价与开盘价比率排名
	factor_data['gtja_alpha111'] = cross_sectional_rank(factor_data['_ts_close_open_ratio'])
	
	# Alpha112: RANK(HIGH / LOW)
	# 含义：最高价与最低价比率排名
	factor_data['gtja_alpha112'] = cross_sectional_rank(factor_data['_ts_high_low_ratio'])
	
	# Alpha113: RANK(VOLUME / ADV20)
	# 含义：成交量与平均成交量比率排名
	factor_data['gtja_alpha113'] = cross_sectional_rank(vol / (adv20 + 1e-10))
	
	# Alpha114: RANK(VOLUME / ADV10)
	# 含义：成交量与平均成交量比率排名
	factor_data['gtja_alpha114'] = cross_sectional_rank(vol / (factor_data['_ts_adv10'] + 1e-10))
	
	# Alpha115: (-1 * RANK(CORR(CLOSE, VOLUME, 5)))
	# 含义：收盘价与成交量相关性排名
	factor_data['gtja_alpha115'] = -1 * cross_sectional_rank(ts_corr(close, vol, 5))
	
	# Alpha116: (-1 * RANK(CORR(CLOSE, VOLUME, 10)))
	# 含义：收盘价与成交量相关性排名
	factor_data['gtja_alpha116'] = -1 * cross_sectional_rank(ts_corr(close, vol, 10))
	
	# Alpha117: (-1 * RANK(CORR(OPEN, VOLUME, 5)))
	# 含义：开盘价与成交量相关性排名
	factor_data['gtja_alpha117'] = -1 * cross_sectional_rank(ts_corr(open_price, vol, 5))
	
	# Alpha118: (-1 * RANK(CORR(OPEN, VOLUME, 10)))
	# 含义：开盘价与成交量相关性排名
	factor_data['gtja_alpha118'] = -1 * cross_sectional_rank(ts_corr(open_price, vol, 10))
	
	# Alpha119: (-1 * RANK(CORR(HIGH, VOLUME, 5)))
	# 含义：最高价与成交量相关性排名
	factor_data['gtja_alpha119'] = -1 * cross_sectional_rank(ts_corr(high, vol, 5))
	
	# Alpha120: (-1 * RANK(CORR(HIGH, VOLUME, 10)))
	# 含义：最高价与成交量相关性排名
	factor_data['gtja_alpha120'] = -1 * cross_sectional_rank(ts_corr(high, vol, 10))
	
	# Alpha121: (-1 * RANK(CORR(LOW, VOLUME, 5)))
	# 含义：最低价与成交量相关性排名
	factor_data['gtja_alpha121'] = -1 * cross_sectional_rank(ts_corr(low, vol, 5))
	
	# Alpha122: (-1 * RANK(CORR(LOW, VOLUME, 10)))
	# 含义：最低价与成交量相关性排名
	factor_data['gtja_alpha122'] = -1 * cross_sectional_rank(ts_corr(low, vol, 10))
	
	# Alpha123: (-1 * RANK(CORR(VWAP, VOLUME, 5)))
	# 含义：VWAP与成交量相关性排名
	factor_data['gtja_alpha123'] = -1 * cross_sectional_rank(ts_corr(vwap, vol, 5))
	
	# Alpha124: (-1 * RANK(CORR(VWAP, VOLUME, 10)))
	# 含义：VWAP与成交量相关性排名
	factor_data['gtja_alpha124'] = -1 * cross_sectional_rank(ts_corr(vwap, vol, 10))
	
	# Alpha125: (-1 * RANK(CORR(CLOSE, OPEN, 5)))
	# 含义：收盘价与开盘价相关性排名
	factor_data['gtja_alpha125'] = -1 * cross_sectional_rank(ts_corr(close, open_price, 5))
	
	# Alpha126: (-1 * RANK(CORR(CLOSE, OPEN, 10)))
	# 含义：收盘价与开盘价相关性排名
	factor_data['gtja_alpha126'] = -1 * cross_sectional_rank(ts_corr(close, open_price, 10))
	
	# Alpha127: (-1 * RANK(CORR(HIGH, LOW, 5)))
	# 含义：最高价与最低价相关性排名
	factor_data['gtja_alpha127'] = -1 * cross_sectional_rank(ts_corr(high, low, 5))
	
	# Alpha128: (-1 * RANK(CORR(HIGH, LOW, 10)))
	# 含义：最高价与最低价相关性排名
	factor_data['gtja_alpha128'] = -1 * cross_sectional_rank(ts_corr(high, low, 10))
	
	# Alpha129: (-1 * RANK(CORR(CLOSE, VWAP, 5)))
	# 含义：收盘价与VWAP相关性排名
	factor_data['gtja_alpha129'] = -1 * cross_sectional_rank(ts_corr(close, vwap, 5))
	
	# Alpha130: (-1 * RANK(CORR(CLOSE, VWAP, 10)))
	# 含义：收盘价与VWAP相关性排名
	factor_data['gtja_alpha130'] = -1 * cross_sectional_rank(ts_corr(close, vwap, 10))
	
	# Alpha131: (-1 * RANK(CORR(OPEN, VWAP, 5)))
	# 含义：开盘价与VWAP相关性排名
	factor_data['gtja_alpha131'] = -1 * cross_sectional_rank(ts_corr(open_price, vwap, 5))
	
	# Alpha132: (-1 * RANK(CORR(OPEN, VWAP, 10)))
	# 含义：开盘价与VWAP相关性排名
	factor_data['gtja_alpha132'] = -1 * cross_sectional_rank(ts_corr(open_price, vwap, 10))
	
	# Alpha133: (-1 * RANK(CORR(HIGH, VWAP, 5)))
	# 含义：最高价与VWAP相关性排名
	factor_data['gtja_alpha133'] = -1 * cross_sectional_rank(ts_corr(high, vwap, 5))
	
	# Alpha134: (-1 * RANK(CORR(HIGH, VWAP, 10)))
	# 含义：最高价与VWAP相关性排名
	factor_data['gtja_alpha134'] = -1 * cross_sectional_rank(ts_corr(high, vwap, 10))
	
	# Alpha135: (-1 * RANK(CORR(LOW, VWAP, 5)))
	# 含义：最低价与VWAP相关性排名
	factor_data['gtja_alpha135'] = -1 * cross_sectional_rank(ts_corr(low, vwap, 5))
	
	# Alpha136: (-1 * RANK(CORR(LOW, VWAP, 10)))
	# 含义：最低价与VWAP相关性排名
	factor_data['gtja_alpha136'] = -1 * cross_sectional_rank(ts_corr(low, vwap, 10))
	
	# Alpha137: (-1 * RANK(CORR(RETURNS, VOLUME, 5)))
	# 含义：收益率与成交量相关性排名
	factor_data['gtja_alpha137'] = -1 * cross_sectional_rank(ts_corr(returns, vol, 5))
	
	# Alpha138: (-1 * RANK(CORR(RETURNS, VOLUME, 10)))
	# 含义：收益率与成交量相关性排名
	factor_data['gtja_alpha138'] = -1 * cross_sectional_rank(ts_corr(returns, vol, 10))
	
	# Alpha139: (-1 * RANK(STDDEV(CLOSE, 3)))
	# 含义：收盘价波动性排名
	factor_data['gtja_alpha139'] = -1 * cross_sectional_rank(factor_data['_ts_std_close_3'])
	
	# Alpha140: (-1 * RANK(STDDEV(CLOSE, 20)))
	# 含义：收盘价波动性排名
	factor_data['gtja_alpha140'] = -1 * cross_sectional_rank(factor_data['_ts_std_close_20'])
	
	# Alpha141: (-1 * RANK(STDDEV(OPEN, 5)))
	# 含义：开盘价波动性排名
	factor_data['gtja_alpha141'] = -1 * cross_sectional_rank(factor_data['_ts_std_open_5'])
	
	# Alpha142: (-1 * RANK(STDDEV(OPEN, 10)))
	# 含义：开盘价波动性排名
	factor_data['gtja_alpha142'] = -1 * cross_sectional_rank(factor_data['_ts_std_open_10'])
	
	# Alpha143: (-1 * RANK(STDDEV(HIGH, 5)))
	# 含义：最高价波动性排名
	factor_data['gtja_alpha143'] = -1 * cross_sectional_rank(factor_data['_ts_std_high_5'])
	
	# Alpha144: (-1 * RANK(STDDEV(HIGH, 10)))
	# 含义：最高价波动性排名
	factor_data['gtja_alpha144'] = -1 * cross_sectional_rank(factor_data['_ts_std_high_10'])
	
	# Alpha145: (-1 * RANK(STDDEV(LOW, 5)))
	# 含义：最低价波动性排名
	factor_data['gtja_alpha145'] = -1 * cross_sectional_rank(factor_data['_ts_std_low_5'])
	
	# Alpha146: (-1 * RANK(STDDEV(LOW, 10)))
	# 含义：最低价波动性排名
	factor_data['gtja_alpha146'] = -1 * cross_sectional_rank(factor_data['_ts_std_low_10'])
	
	# Alpha147: (-1 * RANK(STDDEV(VWAP, 5)))
	# 含义：VWAP波动性排名
	factor_data['gtja_alpha147'] = -1 * cross_sectional_rank(factor_data['_ts_std_vwap_5'])
	
	# Alpha148: (-1 * RANK(STDDEV(VWAP, 10)))
	# 含义：VWAP波动性排名
	factor_data['gtja_alpha148'] = -1 * cross_sectional_rank(factor_data['_ts_std_vwap_10'])
	
	# Alpha149: (-1 * RANK(STDDEV(VOLUME, 3)))
	# 含义：成交量波动性排名
	factor_data['gtja_alpha149'] = -1 * cross_sectional_rank(factor_data['_ts_std_vol_5'])
	
	# Alpha150: (-1 * RANK(STDDEV(VOLUME, 20)))
	# 含义：成交量波动性排名
	factor_data['gtja_alpha150'] = -1 * cross_sectional_rank(factor_data['_ts_std_vol_20'])
	
	# ==================== Alpha151-Alpha191 ====================
	# Alpha151: (-1 * RANK(STDDEV(RETURNS, 3)))
	# 含义：收益率波动性排名
	factor_data['gtja_alpha151'] = -1 * cross_sectional_rank(factor_data['_ts_std_returns_3'])
	
	# Alpha152: (-1 * RANK(STDDEV(RETURNS, 20)))
	# 含义：收益率波动性排名
	factor_data['gtja_alpha152'] = -1 * cross_sectional_rank(factor_data['_ts_std_returns_20'])
	
	# Alpha153: (-1 * RANK(SUM(RETURNS, 3)))
	# 含义：收益率累积排名
	factor_data['gtja_alpha153'] = -1 * cross_sectional_rank(factor_data['_ts_sum_returns_3'])
	
	# Alpha154: (-1 * RANK(SUM(RETURNS, 20)))
	# 含义：收益率累积排名
	factor_data['gtja_alpha154'] = -1 * cross_sectional_rank(factor_data['_ts_sum_returns_20'])
	
	# Alpha155: (-1 * RANK(DELTA(CLOSE, 10)))
	# 含义：价格变化排名
	factor_data['gtja_alpha155'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_10'])
	
	# Alpha156: (-1 * RANK(DELTA(VOLUME, 10)))
	# 含义：成交量变化排名
	factor_data['gtja_alpha156'] = -1 * cross_sectional_rank(factor_data['_ts_delta_vol_10'])
	
	# Alpha157: (-1 * RANK(TS_RANK(CLOSE, 20)))
	# 含义：收盘价时间序列排名
	factor_data['gtja_alpha157'] = -1 * cross_sectional_rank(ts_rank(close, 20))
	
	# Alpha158: (-1 * RANK(TS_RANK(VOLUME, 20)))
	# 含义：成交量时间序列排名
	factor_data['gtja_alpha158'] = -1 * cross_sectional_rank(ts_rank(vol, 20))
	
	# Alpha159: (-1 * RANK(TS_RANK(OPEN, 20)))
	# 含义：开盘价时间序列排名
	factor_data['gtja_alpha159'] = -1 * cross_sectional_rank(ts_rank(open_price, 20))
	
	# Alpha160: (-1 * RANK(TS_RANK(HIGH, 20)))
	# 含义：最高价时间序列排名
	factor_data['gtja_alpha160'] = -1 * cross_sectional_rank(ts_rank(high, 20))
	
	# Alpha161: (-1 * RANK(TS_RANK(LOW, 20)))
	# 含义：最低价时间序列排名
	factor_data['gtja_alpha161'] = -1 * cross_sectional_rank(ts_rank(low, 20))
	
	# Alpha162: (-1 * RANK(TS_RANK(VWAP, 20)))
	# 含义：VWAP时间序列排名
	factor_data['gtja_alpha162'] = -1 * cross_sectional_rank(ts_rank(vwap, 20))
	
	# Alpha163: (-1 * RANK(CORR(CLOSE, VOLUME, 3)))
	# 含义：收盘价与成交量相关性排名
	factor_data['gtja_alpha163'] = -1 * cross_sectional_rank(ts_corr(close, vol, 3))
	
	# Alpha164: (-1 * RANK(CORR(OPEN, VOLUME, 3)))
	# 含义：开盘价与成交量相关性排名
	factor_data['gtja_alpha164'] = -1 * cross_sectional_rank(ts_corr(open_price, vol, 3))
	
	# Alpha165: (-1 * RANK(CORR(HIGH, VOLUME, 3)))
	# 含义：最高价与成交量相关性排名
	factor_data['gtja_alpha165'] = -1 * cross_sectional_rank(ts_corr(high, vol, 3))
	
	# Alpha166: (-1 * RANK(CORR(LOW, VOLUME, 3)))
	# 含义：最低价与成交量相关性排名
	factor_data['gtja_alpha166'] = -1 * cross_sectional_rank(ts_corr(low, vol, 3))
	
	# Alpha167: (-1 * RANK(CORR(VWAP, VOLUME, 3)))
	# 含义：VWAP与成交量相关性排名
	factor_data['gtja_alpha167'] = -1 * cross_sectional_rank(ts_corr(vwap, vol, 3))
	
	# Alpha168: (-1 * RANK(CORR(CLOSE, OPEN, 3)))
	# 含义：收盘价与开盘价相关性排名
	factor_data['gtja_alpha168'] = -1 * cross_sectional_rank(ts_corr(close, open_price, 3))
	
	# Alpha169: (-1 * RANK(CORR(HIGH, LOW, 3)))
	# 含义：最高价与最低价相关性排名
	factor_data['gtja_alpha169'] = -1 * cross_sectional_rank(ts_corr(high, low, 3))
	
	# Alpha170: (-1 * RANK(CORR(CLOSE, VWAP, 3)))
	# 含义：收盘价与VWAP相关性排名
	factor_data['gtja_alpha170'] = -1 * cross_sectional_rank(ts_corr(close, vwap, 3))
	
	# Alpha171: (-1 * RANK(CORR(OPEN, VWAP, 3)))
	# 含义：开盘价与VWAP相关性排名
	factor_data['gtja_alpha171'] = -1 * cross_sectional_rank(ts_corr(open_price, vwap, 3))
	
	# Alpha172: (-1 * RANK(CORR(HIGH, VWAP, 3)))
	# 含义：最高价与VWAP相关性排名
	factor_data['gtja_alpha172'] = -1 * cross_sectional_rank(ts_corr(high, vwap, 3))
	
	# Alpha173: (-1 * RANK(CORR(LOW, VWAP, 3)))
	# 含义：最低价与VWAP相关性排名
	factor_data['gtja_alpha173'] = -1 * cross_sectional_rank(ts_corr(low, vwap, 3))
	
	# Alpha174: (-1 * RANK(CORR(RETURNS, VOLUME, 3)))
	# 含义：收益率与成交量相关性排名
	factor_data['gtja_alpha174'] = -1 * cross_sectional_rank(ts_corr(returns, vol, 3))
	
	# Alpha175: (-1 * RANK(CORR(CLOSE, VOLUME, 20)))
	# 含义：收盘价与成交量相关性排名
	factor_data['gtja_alpha175'] = -1 * cross_sectional_rank(ts_corr(close, vol, 20))
	
	# Alpha176: (-1 * RANK(CORR(OPEN, VOLUME, 20)))
	# 含义：开盘价与成交量相关性排名
	factor_data['gtja_alpha176'] = -1 * cross_sectional_rank(ts_corr(open_price, vol, 20))
	
	# Alpha177: (-1 * RANK(CORR(HIGH, VOLUME, 20)))
	# 含义：最高价与成交量相关性排名
	factor_data['gtja_alpha177'] = -1 * cross_sectional_rank(ts_corr(high, vol, 20))
	
	# Alpha178: (-1 * RANK(CORR(LOW, VOLUME, 20)))
	# 含义：最低价与成交量相关性排名
	factor_data['gtja_alpha178'] = -1 * cross_sectional_rank(ts_corr(low, vol, 20))
	
	# Alpha179: (-1 * RANK(CORR(VWAP, VOLUME, 20)))
	# 含义：VWAP与成交量相关性排名
	factor_data['gtja_alpha179'] = -1 * cross_sectional_rank(ts_corr(vwap, vol, 20))
	
	# Alpha180: (-1 * RANK(CORR(CLOSE, OPEN, 20)))
	# 含义：收盘价与开盘价相关性排名
	factor_data['gtja_alpha180'] = -1 * cross_sectional_rank(ts_corr(close, open_price, 20))
	
	# Alpha181: (-1 * RANK(CORR(HIGH, LOW, 20)))
	# 含义：最高价与最低价相关性排名
	factor_data['gtja_alpha181'] = -1 * cross_sectional_rank(ts_corr(high, low, 20))
	
	# Alpha182: (-1 * RANK(CORR(CLOSE, VWAP, 20)))
	# 含义：收盘价与VWAP相关性排名
	factor_data['gtja_alpha182'] = -1 * cross_sectional_rank(ts_corr(close, vwap, 20))
	
	# Alpha183: (-1 * RANK(CORR(OPEN, VWAP, 20)))
	# 含义：开盘价与VWAP相关性排名
	factor_data['gtja_alpha183'] = -1 * cross_sectional_rank(ts_corr(open_price, vwap, 20))
	
	# Alpha184: (-1 * RANK(CORR(HIGH, VWAP, 20)))
	# 含义：最高价与VWAP相关性排名
	factor_data['gtja_alpha184'] = -1 * cross_sectional_rank(ts_corr(high, vwap, 20))
	
	# Alpha185: (-1 * RANK(CORR(LOW, VWAP, 20)))
	# 含义：最低价与VWAP相关性排名
	factor_data['gtja_alpha185'] = -1 * cross_sectional_rank(ts_corr(low, vwap, 20))
	
	# Alpha186: (-1 * RANK(CORR(RETURNS, VOLUME, 20)))
	# 含义：收益率与成交量相关性排名
	factor_data['gtja_alpha186'] = -1 * cross_sectional_rank(ts_corr(returns, vol, 20))
	
	# Alpha187: (-1 * RANK(DELTA(LOG(VOLUME), 2)))
	# 含义：对数成交量变化排名
	factor_data['gtja_alpha187'] = -1 * cross_sectional_rank(factor_data['_ts_delta_log_vol_2'])
	
	# Alpha188: (-1 * RANK(CLOSE - DELAY(CLOSE, 1)))
	# 含义：价格变化排名（与Alpha039相同）
	factor_data['gtja_alpha188'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_1'])
	
	# Alpha189: (-1 * RANK(CLOSE - DELAY(CLOSE, 2)))
	# 含义：价格变化排名（与Alpha040相同）
	factor_data['gtja_alpha189'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_2'])
	
	# Alpha190: (-1 * RANK(CLOSE - DELAY(CLOSE, 3)))
	# 含义：价格变化排名（与Alpha041相同）
	factor_data['gtja_alpha190'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_3'])
	
	# Alpha191: (-1 * RANK(CLOSE - DELAY(CLOSE, 5)))
	# 含义：价格变化排名（与Alpha042相同）
	factor_data['gtja_alpha191'] = -1 * cross_sectional_rank(factor_data['_ts_delta_close_5'])
	
	# 删除中间变量列（以_ts_开头的列）
	ts_cols = [col for col in factor_data.columns if col.startswith('_ts_')]
	factor_data = factor_data.drop(columns=ts_cols)
	
	# 编码股票代码
	factor_data['stock_encode'] = stock_enc.fit_transform(factor_data['stock_code'])
	# 填充行业缺失值
	factor_data['industry'] = factor_data['industry'].fillna('')
	# 编码行业
	factor_data['industry_encode'] = industry_enc.fit_transform(factor_data['industry'])
	# 编码企业类型
	factor_data['act_ent_type_encode'] = act_ent_type_enc.fit_transform(factor_data['act_ent_type'])
	
	# 返回因子数据
	return factor_data


def download_cdb_data(down_db, data_file_url):
	"""下载并处理数据仓库数据

	Args:
		down_db: 是否下载到数据库
		
	Returns:
		pd.DataFrame: 处理后的因子数据
	"""
	# 记录模块启动日志
	logging.info(f'#--------------------------------------------数据加工模块启动--------------------------------------------#')
	# 获取数据库引擎
	# engine = get_sql_engine('cdb')
	# 记录数据库连接成功日志
	logging.info(f'数据加工模块：1.MYSQL数仓连接成功')
	# 获取整合数据
	integ_data = get_integ_data(data_file_url)
	
	# 记录数据查询完成日志
	logging.info(f'数据加工模块：2.整合数据查询完成')
	
	# 获取因子数据
	factor_data = get_factor_data(integ_data)
	
	# 记录因子加工完成日志
	logging.info(f'数据加工模块：3.因子加工完成')
	
	
	# 替换无穷值为NaN
	factor_data.replace([np.inf, -np.inf], np.nan, inplace=True)
	# 保存因子数据到parquet文件
	# conn = sqlite3.connect('D:/办公/量化交易/quant_project/data_file/odb.db')
	# factor_data.to_sql('stock_factor_data', con=conn, if_exists='replace', index=False)
	final_path = Path(data_file_url) / 'stock_factor_data.parquet'
	temp_path = final_path.with_suffix('.tmp.parquet')
	factor_data.to_parquet(temp_path, index=False)
	temp_path.replace(final_path)
	# conn.close()

	
	# 记录数据入仓完成日志
	logging.info(f'数据加工模块：4.因子数据入仓完成，表名为stock_factor_data')
	# 记录数据表导出成功日志
	logging.info(f'数据加工模块：5.数据表导出成功 库名：cdb，表名：stock_factor_data')
	# 记录模块完成日志
	logging.info(f'#--------------------------------------------数据加工模块完成--------------------------------------------#')
	
	# 返回因子数据
	return factor_data

if __name__ == '__main__':
	"""主程序入口"""
	# 调用数据加工函数
	data = download_cdb_data(down_db=True, data_file_url='D:/work/quant/quant001/quant/data_file')	
