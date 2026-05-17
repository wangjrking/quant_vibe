import pandas as pd 
import tushare as ts
import xgboost as xgb
from sklearn.metrics import roc_auc_score,accuracy_score, mean_squared_error
from datetime import datetime
from sqlalchemy import text
import numpy as np
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from database_module import get_sql_engine



def get_pro(ts_token):
    ts_pro = ts.pro_api(ts_token)
    return ts_pro 
# ========== 1. 自定义函数扩展（时间序列函数示例） ==========
def _ts_mean(x, window=5):
    """计算滚动均值"""
    return pd.Series(x.flatten()).rolling(window).mean().values.reshape(-1, 1)

def _ts_std(x, window=5):
    """计算滚动标准差"""
    return pd.Series(x.flatten()).rolling(window).std().values.reshape(-1, 1)

def get_model(function_lst):
    
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']
    # function_set = function_set + function_lst
    gp_transformer = SymbolicTransformer(
    generations=50, # 进化迭代的代数，值越大搜索空间越广但耗时增加
    population_size=5000, # 每代种群中个体（公式树）的数量，影响搜索多样性
    hall_of_fame=100, # 保留历史最优个体的数量，用于后续特征生成
    n_components=5,       # 最终生成的新特征数量
    stopping_criteria=0.0001, # 适应度阈值，达到后提前终止进化（如设为0.01表示MSE低于0.01时停止）
    function_set=function_set, # 允许的数学运算符集合，支持加减乘除、函数等
    init_depth=(2,6), # 公式树的初始深度范围（(min_depth, max_depth)）
    p_crossover=0.9, # 交叉变异概率（子树交换）
    p_subtree_mutation=0.01, # 子树变异概率（替换整棵子树）
    p_hoist_mutation=0.01, # 抬升变异概率（提升子树层级以简化公式）
    p_point_mutation=0.01, # 点变异概率（修改单个节点）
    parsimony_coefficient=0.001, # 公式复杂度惩罚系数，值越大倾向于生成更简单的公式
    metric='spearman', # 适应度评估指标（如 'pearson'、'spearman' 或自定义函数）
    max_samples=0.9, # 用于适应度计算的样本比例（0.0-1.0），可加速训练
    n_jobs=-1, # 并行计算线程数，-1表示使用所有CPU核心
    verbose=1, # 日志输出级别（0静默，1显示进度）
    random_state=42
)
    return gp_transformer



def get_factor_data(data_test_dt, engine, label):

    factor_data = pd.read_parquet('D:/办公\量化交易/quant_project/data_file/cdb/factor_data.parquet')
    factor_data = factor_data[~factor_data['name'].str.contains('ST')]
    

    # 使用 np.select 计算收益率

    train_data = factor_data[factor_data['trade_date'] < data_test_dt]
    

    test_data = factor_data[factor_data['trade_date'] >= data_test_dt]
    train_data = train_data.dropna(subset=[label]) 

    factor_list = [
    'close','open','high','low','vol','amount',
    'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
    'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv',  label
    ]
    
    
    train_factor_data = train_data[factor_list]
    test_factor_data = test_data[factor_list]
    train_factor_data = train_factor_data.dropna()
    test_factor_data = test_factor_data.dropna()

    train_factor_data = train_factor_data.apply(pd.to_numeric, errors='coerce')
    test_factor_data = test_factor_data.apply(pd.to_numeric, errors='coerce')
    
    
    train_y = train_factor_data.loc[:, label]
    train_x = train_factor_data.drop(columns=[label]) 
    test_y = test_factor_data.loc[:, label]
    test_x = test_factor_data.drop(columns=[label]) 
    
    return train_x, train_y, test_x, test_y, train_data, test_data
    
def download_pred_data(data, engine, label): 
    data.to_sql('stock_predict_data_'+label, engine, if_exists='replace', index=False)
    data.to_parquet('D:/办公\量化交易/quant_project/data_file/cdb/stock_predict_data_'+label+'.parquet')

def download_pdb_data( data_test_dt, label, type):
    
    print(f'#--------------------------------------------AI模块启动--------------------------------------------#')
    
    engine = get_sql_engine('pdb')
    
    print(f'AI模块：1.MYSQL数据库连接成功')
    train_x, train_y, test_x, test_y, train_data, test_data = get_factor_data( data_test_dt, engine, label)
    
    function_lst = [_ts_mean,_ts_std]
    
    model = get_model(function_lst)
    print(train_x,train_y)
    x_new = model.fit_transform(train_x, train_y)  # 再转换
    print(x_new)

    for i, program in enumerate(model._best_programs):
        if program is not None:
            print(f"特征 {i+1}: {str(program)}")

    # ========== 5. 可视化结果 ==========
    
    print(f'AI模块：3.模型建立完成')
    
    print(f'#--------------------------------------------AI模块完成--------------------------------------------#')


if __name__ == '__main__':
    data_test_dt = '20250619'
    download_pdb_data(data_test_dt, label='yield_rate', type='reg')

    # download_pdb_data( data_test_dt, label='tag', type='class')


    
    