import pandas as pd
import tushare as ts
import xgboost as xgb
from sklearn.metrics import roc_auc_score,accuracy_score, mean_squared_error
from datetime import datetime
from sqlalchemy import text
import numpy as np
import alphalens
import matplotlib.pyplot as plt
import pickle
import shap
from pathlib import Path
from project_paths import resolve_data_dir

from database_module import get_sql_engine

def get_factor_def():
    engine = get_sql_engine('cdb')
    query = text('''
    SELECT *
    FROM CDB.factor_define
    ''')
    factor_data = pd.read_sql(query, engine)
    return factor_data


def save_factor_def(data):
    engine = get_sql_engine('cdb')
    data.to_sql('factor_define', engine, if_exists='replace', index=False)


def get_factor_stock_data(factor_data,factor,groupby_lst):
    factor_data['trade_date'] = pd.to_datetime(factor_data['trade_date'], format='%Y%m%d').dt.tz_localize('Asia/Shanghai')
    factor_data = factor_data.dropna(subset=[factor])
    multi_factor_data = factor_data[factor]
    multi_factor_data.index = factor_data.set_index(['trade_date', 'stock_code']).index

    groupby_data = factor_data[groupby_lst]
    groupby_data.index = factor_data.set_index(['trade_date', 'stock_code']).index

    stock_price_data  = factor_data.pivot(
    index='trade_date',  # 行索引
    columns='stock_code',  # 列名
    values='close'       # 填充的值
)
    multi_factor_data.to_csv('test.csv')
    alpha_factor_data = alphalens.utils.get_clean_factor_and_forward_returns(multi_factor_data, # 因子数据
                                                                   stock_price_data,  # 股票价格数据
                                                                   quantiles=10,  # 股票分组
                                                                   # bins=5,  # 因子分组
                                                                   groupby=groupby_data,
                                                                    periods=(1, 2, 10)
                                                                    )  # 行业分组
    return alpha_factor_data

def get_factor_analysis_data(factor, groupby='industry'):

    factor_data = pd.read_parquet(resolve_data_dir() / 'stock_predict_data_5d_yield_rate.parquet')

    alpha_factor_data = get_factor_stock_data(factor_data,factor, groupby)
    alphalens.tears.create_full_tear_sheet(alpha_factor_data)

def _resolve_shap_artifact_dir(artifact_dir=None) -> Path:
    candidates = []
    if artifact_dir is not None:
        candidates.append(Path(artifact_dir))

    env_dir = os.getenv("QUANT_SHAP_ARTIFACT_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    data_dir = resolve_data_dir()
    candidates.extend(
        [
            data_dir / "reports" / "model_explainability_current",
        ]
    )

    for candidate in candidates:
        shap_path = candidate / "shap_values.pkl"
        index_path = candidate / "test_index.pkl"
        if shap_path.exists() and index_path.exists():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "shap artifact files not found. expected shap_values.pkl and test_index.pkl in one of:\n"
        + searched
    )


def shap_explain(stock_code,trade_date, artifact_dir=None):
    artifact_root = _resolve_shap_artifact_dir(artifact_dir)

    with open(artifact_root / 'shap_values.pkl', 'rb') as file:
        shap_values  = pickle.load(file)
    with open(artifact_root / 'test_index.pkl', 'rb') as file:
        test_index  = pickle.load(file)
    num = test_index.index((stock_code,trade_date))
    shap.plots.waterfall(shap_values[num])



if __name__ == '__main__':

    factor = 'pred_prob'
    factor_data = pd.read_parquet(resolve_data_dir() / 'stock_predict_data_10d_yield_rate.parquet')
    shap_explain(stock_code='002999.SZ',trade_date='20260210')
