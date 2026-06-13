# 流动性精选一号

`prod_liq_prime_one_v20260612` 是投产策略，来源版本为 `liq_enh_top1_q95_amt8e5_turn2_mv1e6`。

## 一句话定义

基于 `executable_5d_open_return` 预测标签，使用 2010 年以来滚动/扩展训练集训练 XGBoost 模型，每期按 IC 筛选 Top120 因子，并在预测结果中按 q95 分位、成交额不低于 80 万、换手率不低于 2、总市值不高于 100 万口径筛选后选取预测分最高的 1 只股票作为交易信号。

## 买入信号

在每日预测结果中，股票的 `executable_5d_open_return` 预测分进入全市场前 5%（q95），且满足成交额不低于 80 万、换手率不低于 2、总市值不高于 100 万口径等流动性过滤条件后，选取预测分最高的 1 只股票，于下一交易日开盘买入。

## 卖出信号

买入后按 5 个交易日持有周期执行退出；掘金实盘/回测环境中同时遵循策略脚本内的持仓、风控和可交易性检查。

## 回测效果

掘金回测累计收益约 1152.09%，年化收益约 385.08%，最大回撤约 28.64%，夏普比率 1.31，Calmar 比率 13.45；开仓 102 次、平仓 101 次，胜率约 61.39%。

本地同规则流动性网格累计收益约 829.78%，年化收益约 215.53%，夏普约 1.90，交易 110 次，胜率约 51.82%。

## 最新信号快照

| signal_date | buy_date | stock_code | name | pred_prob | pred_pct |
| --- | --- | --- | --- | ---: | ---: |
| 20260611 | 20260612 | 301313.SZ | 凡拓数创 | 0.1161055341 | 0.9933481153 |

## 证据来源

- `data_file/reports/open_daily_tuning_20260612/juejin_prod_aligned_4y_liq_enh_top1_q95_amt8e5_turn2_mv1e6_slip0015_cash60w.log`
- `data_file/reports/open_daily_tuning_20260612/rolling_exec5d_prod_rebuild_aligned_early_202206_202406_summary.csv`
- `data_file/reports/open_daily_tuning_20260612/rolling_exec5d_prod_rebuild_aligned_early_202206_202406_foldfs120/selected_features_executable_5d_open_return_rolling_fold8.json`
- `data_file/reports/liq_enh_signal_20260611_for_20260612.json`
