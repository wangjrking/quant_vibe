# 导入数据加载模块，用于下载基础数据和获取Tushare pro接口
from data_load_module import download_odb_data, get_pro

# 导入数据库模块，用于执行SQL语句
from database_module import run_sql, sql

# 导入数据处理模块，用于下载处理后的数据
from data_process_module import download_cdb_data

# 导入AI模块，用于下载AI预测数据
from ai_module import download_pdb_data

# 导入日期时间处理模块
from datetime import datetime
# 导入数据分析库
import pandas as pd
# 导入日志模块
import logging
# 导入JSON处理模块
import json
# 导入Tushare金融数据接口
import tushare as ts
import os


class Run:
	"""量化交易项目主运行类"""
	
	def __init__(self):
		"""初始化方法，加载配置、设置日志并运行项目"""

		# 加载默认配置文件
		self.config = self._load_default_config()

		# 注入临时配置参数
		self.config['custom'] = {
			'data_start_dt':'20150101',  # 数据开始日期
			'data_test_dt':'20250101',   # 测试数据开始日期
			'data_end_dt':datetime.now().strftime("%Y%m%d"),  # 数据结束日期（当前日期）
			'predict_label':'adjust_10d_yield_rate'  # 预测标签：10日收益率
		}

		# 设定日志配置
		self._setup_logging()

		# 运行项目主流程                                                            
		self._run_project() 

	def _load_default_config(self) -> dict:
		"""加载默认配置文件
		
		Returns:
			dict: 配置信息字典
		"""
		with open('./config.json', 'r', encoding='utf-8') as f:
			config = json.load(f)
		return config
	
	def _setup_logging(self):
		"""设置日志配置"""
		# 获取数据结束日期，用于生成日志文件名
		data_end_dt = self.config['custom']['data_end_dt']

		# 构建日志文件路径
		self.log_file_url = self.config['filecatalog']['file_url'] + "/log"
		os.makedirs(self.log_file_url, exist_ok=True)
		
		self.data_file_url = self.config['filecatalog']['file_url'] + "/data_file"
		os.makedirs(self.data_file_url, exist_ok=True)



		today_log_file_url = self.log_file_url + "/" + data_end_dt


		# 配置日志基本设置
		logging.basicConfig(
			level=logging.INFO,  # 日志器级别设为INFO
			filename=today_log_file_url,   # 输出到文件
			filemode="a",         # 追加模式
			encoding="utf-8",     # 中文编码
			format="%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s",  # 日志格式
			datefmt="%Y-%m-%d %H:%M:%S"  # 时间格式
			)
	
	def get_stock_lst(self):
		"""获取股票代码列表
		
		Returns:
			list: 股票代码列表
		"""
		# 获取数据开始日期
		data_start_dt = self.config['custom']['data_start_dt']

		# 获取TUSHARE的API token
		ts_token = self.config['datasource']['tushare_token']

		# 获取Tushare pro接口
		print(ts_token)
		ts_pro = get_pro(ts_token)

		
		
		# 初始化股票代码列表
		stock_code_lst = []
		
		# 从指定指数中获取成分股
		# 这里只使用了'932000.CSI'（中证1000指数），其他指数被注释
		for index_id in ['932000.CSI' ]: # 000300.SH 沪深300指数,000905.SH 中证500指数,000852.SH 中证1000指数,932000.CSI 中证2000指数
			# 获取指数成分股并添加到列表
			stock_code_lst += ts_pro.index_weight(index_code=index_id,trade_date='20251231')['con_code'].tolist()
		stock_code_lst = [s for s in stock_code_lst if not s.endswith(".BJ")]
		# 取所有A股
		# stock_data = ts_pro.query('stock_basic', list_status='L', fields='ts_code,list_date')
		# stock_code_lst = stock_data['ts_code'].tolist()
		


		# 以下代码被注释，用于测试
		# stock_code_lst = ['000863.SZ','000036.SZ']
		
		# 返回股票代码列表
		return stock_code_lst

	def _run_project(self):
		"""运行项目主流程"""
		# 从配置中获取日期参数
		data_start_dt = self.config['custom']['data_start_dt']
		data_test_dt = self.config['custom']['data_test_dt']
		data_end_dt = self.config['custom']['data_end_dt']
		predict_label = self.config['custom']['predict_label']

		# 获取Tushare API token
		ts_token = self.config['datasource']['tushare_token']
		# 以下代码被注释，原本用于直接获取Tushare pro接口
		# ts_pro = get_pro(ts_token)

		# 获取股票代码列表
		stock_code_lst = self.get_stock_lst()
		
		# 数据载入模块：下载原始基础数据
		download_odb_data(stock_code_lst, data_start_dt, data_end_dt, ts_token, self.data_file_url)

		# 数据仓库模块：执行SQL语句处理数据
		
		run_sql(sql, data_file_url=self.data_file_url)

		# 数据加工模块：下载加工后的数据
		
		download_cdb_data(down_db=True, data_file_url=self.data_file_url)

		# AI模块：下载AI预测数据，type='reg'表示回归任务
		
		download_pdb_data(data_start_dt, data_test_dt, label=predict_label, type='reg', data_file_url=self.data_file_url)
		


if __name__ == '__main__':
	"""主程序入口"""
	# 创建Run实例，启动整个项目
	Run()
 