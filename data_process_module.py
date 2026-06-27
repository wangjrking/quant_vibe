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

from gtja_alpha_official_core import append_official_gtja_alpha
from project_paths import resolve_data_dir
from stock_daily_data_route import connect_stock_daily_readonly




# 数据整合
def get_integ_data(data_file_url):
	conn = connect_stock_daily_readonly(data_dir=data_file_url)
	try:
		integ_data = pd.read_sql("SELECT * FROM STOCK_DAILY_DATA order by stock_code, trade_date", conn)
		# integ_data = conn.execute("SELECT * FROM STOCK_DAILY_DATA order by stock_code, trade_date").df()
	finally:
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

	# 鎸夊畼鏂规埅闈㈠叕寮忕粺涓€璁＄畻 GTJA Alpha191 鍥犲瓙
	factor_data = append_official_gtja_alpha(
		factor_data,
		cross_sectional_rank_mode="rank",
		benchmark_prefix="index_2000",
	)

	# 鍒犻櫎涓棿鍙橀噺鍒楋紙浠ts_寮€澶寸殑鍒楋級
	ts_cols = [col for col in factor_data.columns if col.startswith('_ts_')]
	factor_data = factor_data.drop(columns=ts_cols)

	# 缂栫爜鑲＄エ浠ｇ爜
	factor_data['stock_encode'] = stock_enc.fit_transform(factor_data['stock_code'])
	# 濉厖琛屼笟缂哄け鍊?
	factor_data['industry'] = factor_data['industry'].fillna('')
	# 缂栫爜琛屼笟
	factor_data['industry_encode'] = industry_enc.fit_transform(factor_data['industry'])
	# 缂栫爜浼佷笟绫诲瀷
	factor_data['act_ent_type_encode'] = act_ent_type_enc.fit_transform(factor_data['act_ent_type'])

	# 杩斿洖鍥犲瓙鏁版嵁
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
	# conn = sqlite3.connect(Path(data_file_url) / 'odb.db')
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
	data = download_cdb_data(down_db=True, data_file_url=str(resolve_data_dir()))
