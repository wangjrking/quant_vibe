import pandas as pd
import numpy as np
import logging
from datetime import datetime

from dl_model_module import add_fttransformer_features_simple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="D:/办公/量化交易/quant_project/data_file/ftt.log",
    filemode="a",
    encoding="utf-8"
)

def sort_within_group(group):
    rank_data = group.rank(method='min', ascending=False)
    rank_data_min = rank_data.min()
    rank_data_max = rank_data.max()
    return (rank_data_max-rank_data)/(rank_data_max-rank_data_min)


if __name__ == '__main__':
    data_start_dt = '20200101'
    data_test_dt = '20230101'
    label = '10d_yield_rate'

    logging.info("=" * 50)
    logging.info("FTTRANSFORMER因子提取测试")
    logging.info("=" * 50)

    logging.info("步骤1: 加载数据...")
    factor_data = pd.read_parquet('D:/办公/量化交易/quant_project/data_file/cdb/stock_factor_data.parquet')
    factor_data = factor_data.set_index(['stock_code', 'trade_date'], drop=False)

    factor_data = factor_data[~factor_data['name'].str.contains('ST')]
    factor_data = factor_data[factor_data['st_type'] != 'ST']

    factor_data = factor_data[factor_data['trade_date'] >= data_start_dt]

    factor_data['5d_yield_rate_rank'] = factor_data[['5d_yield_rate']].groupby('trade_date')['5d_yield_rate'].transform(sort_within_group)
    factor_data['open6_yield_rate_rank'] = factor_data[['open6_yield_rate']].groupby('trade_date')['open6_yield_rate'].transform(sort_within_group)
    factor_data['10d_yield_rate_rank'] = factor_data[['10d_yield_rate']].groupby('trade_date')['10d_yield_rate'].transform(sort_within_group)

    logging.info(f"数据加载完成: {factor_data.shape}")

    train_data = factor_data[factor_data['trade_date'] < data_test_dt]
    test_data = factor_data[factor_data['trade_date'] >= data_test_dt]

    train_data = train_data.dropna(subset=[label])

    logging.info(f"训练集大小: {train_data.shape}")
    logging.info(f"测试集大小: {test_data.shape}")

    factor_list = [
        'close_rate', 'open_rate', 'high_rate', 'low_rate', 'vol', 'amount', 'stock_encode',
        'high_open_rate', 'high_close_rate', 'low_open_rate', 'low_close_rate',
        'macd', 'macdsignal', 'macdhist', 'dema', 'ema', 'kama', 'ma', 'midpoint', 'midprice',
        'diff_close_low', 'diff_close_high', 'diff_high_low',
        'std_his_low', 'std_cost_5pct', 'std_cost_15pct', 'std_cost_50pct', 'std_cost_85pct', 'std_cost_95pct',
        'pre_close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pb',
        'total_share', 'float_share', 'total_mv', 'circ_mv',
        'industry_encode',
        'limit_times',
        'open6_yield_rate_rank', '10d_yield_rate_rank'
    ]

    logging.info("步骤2: 提取FTTRANSFORMER特征...")

    train_data_new, test_data_new, model = add_fttransformer_features_simple(
        train_data, test_data, factor_list, label
    )

    logging.info(f"训练集新特征维度: {train_data_new.shape}")
    logging.info(f"测试集新特征维度: {test_data_new.shape}")

    ftt_cols = [col for col in train_data_new.columns if col.startswith('ftt_')]
    logging.info(f"FTTRANSFORMER特征列: {ftt_cols}")

    train_data_new.to_parquet('D:/办公/量化交易/quant_project/data_file/ftt_train_data.parquet')
    test_data_new.to_parquet('D:/办公/量化交易/quant_project/data_file/ftt_test_data.parquet')

    logging.info("特征已保存到 parquet 文件")
    logging.info("=" * 50)
    logging.info("FTTRANSFORMER因子提取完成!")
    logging.info("=" * 50)
    print("Shape:", train_data_new.shape)
    print("\nColumns (first 30):")
    print(train_data_new.columns.tolist()[:30])
    print("\nFTT columns:")
    ftt_cols = [col for col in train_data_new.columns if col.startswith('ftt_')]
    print(ftt_cols)
    print("\nFirst 3 rows:")
    print(train_data_new.head(3))
