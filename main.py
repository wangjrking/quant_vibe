"""Legacy historical mixed-db pipeline entry.

This file keeps the old all-in-one mixed-db workflow for rollback or historical
reproduction only. It is not the current standard project entrypoint.

Current layered workflow references:
- WORKFLOW.md
- quant/main/AGENTS.md

To run this legacy entry intentionally, set:
    QUANT_ALLOW_LEGACY_MAIN=1
"""

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
import sys
from project_paths import config_base_dir, load_config
from stock_pool_module import DEFAULT_INDEX_CODES, fetch_index_stock_pool


LEGACY_ENTRY_NOTICE = """\
main.py 已降级为 legacy / historical mixed-db 入口，默认不再作为当前标准主流程使用。
当前标准分层口径请参见：
- WORKFLOW.md
- quant/main/AGENTS.md

当前默认资产：
- L1 raw split DB: quant/data_file/raw_table_dbs/[table].DB
- L2: quant/data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA
- L3 features: quant/data_file/production_factor_parts/
- L3 labels: quant/data_file/prediction_label_parts/
- L4: 独立预测资产方案，非 odb.db.stock_predict_data_* 默认入口

如需明确运行旧 mixed-db 历史链路，请设置环境变量 QUANT_ALLOW_LEGACY_MAIN=1 后再执行。
"""


class Run:
	"""量化交易项目主运行类"""

	def __init__(self):
		"""初始化方法，加载配置、设置日志并运行项目"""

		# 加载默认配置文件
		self.config = self._load_default_config()

		# 注入临时配置参数nm
		self.config['custom'] = {
			'data_start_dt':'20100101',  # 数据开始日期
			'data_test_dt':'20260101',   # 测试数据开始日期
			'data_end_dt':datetime.now().strftime("%Y%m%d"),  # 数据结束日期（当前日期）
			'predict_label':'10d_yield_rate'  # 预测标签：10日收益率
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
		return load_config()

	def _setup_logging(self):
		"""设置日志配置"""
		# 获取数据结束日期，用于生成日志文件名
		data_end_dt = self.config['custom']['data_end_dt']

		# 构建日志文件路径
		base_dir = config_base_dir(self.config)
		self.log_file_url = base_dir / "log"
		os.makedirs(self.log_file_url, exist_ok=True)

		self.data_file_url = base_dir / "data_file"
		os.makedirs(self.data_file_url, exist_ok=True)



		today_log_file_url = self.log_file_url / data_end_dt


		# 配置日志基本设置
		logging.basicConfig(
			level=logging.INFO,  # 日志器级别设为INFO
			filename=str(today_log_file_url),   # 输出到文件
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
		ts_pro = get_pro(ts_token)



		stock_code_lst = fetch_index_stock_pool(ts_pro, index_codes=DEFAULT_INDEX_CODES, trade_date='20251231')
		# 取所有A股
		# stock_data = ts_pro.query('stock_basic', list_status='L', fields='ts_code,list_date')
		# stock_code_lst = stock_data['ts_code'].tolist()


		# 以下代码被注释，用于测试
		# stock_code_lst = ['300713.SZ','301302.SZ', '301360.SZ']

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
		download_odb_data(stock_code_lst, data_start_dt, data_end_dt, ts_token, str(self.data_file_url))

		# 数据仓库模块：执行SQL语句处理数据

		run_sql(sql, data_file_url=str(self.data_file_url))

		# 数据加工模块：下载加工后的数据

		download_cdb_data(down_db=True, data_file_url=str(self.data_file_url))

		# AI模块：下载AI预测数据，type='reg'表示回归任务

		download_pdb_data(data_start_dt, data_test_dt, label=predict_label, type='reg', data_file_url=str(self.data_file_url))
		# download_pdb_data(data_start_dt, data_test_dt, label='10d_yield_rate', type='reg', data_file_url=self.data_file_url)



if __name__ == '__main__':
	"""Legacy 主程序入口。"""
	if os.environ.get("QUANT_ALLOW_LEGACY_MAIN") != "1":
		print(LEGACY_ENTRY_NOTICE, file=sys.stderr)
		raise SystemExit(2)
	# 创建Run实例，启动整个项目
	Run()
