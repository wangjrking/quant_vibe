# 流动性精选一号 V1.0

`prod_liq_prime_one_v20260612` 是当前投产策略，中文名为“流动性精选一号”，V1.0 固化口径为：`09:30` 先卖、`09:31` 再买，持有期 5 个交易日，评分退出最小持有 2 个交易日。

## 一句话定义

基于 `executable_5d_open_return` 预测标签，使用 2010 年以来滚动/扩展训练集训练 XGBoost 回归模型，每期按训练折内 IC 筛选因子；每日从预测分进入全市场 q95 的股票中，叠加成交额、换手率、总市值和可交易性过滤，选择预测分最高的 1 只股票作为下一交易日买入信号。

## 数据和模型

- 预测标签：`executable_5d_open_return`，表示 T+1 开盘买入、T+6 开盘卖出并扣除训练口径成本后的 5 日可执行收益。
- 预测表：`stock_predict_data_executable_5d_open_return_prod_aligned_202206_202606`。
- 训练方式：rolling / expanding rebuild aligned。
- 训练起点：`20100101`。
- 归档 fold 示例：fold 1 训练 `20100101-20220524`、测试 `20220604-20220903`；fold 8 训练 `20100101-20240222`、测试 `20240304-20240603`。
- 模型：XGBoost regression，参数详情见 `model_params.json`。

## 因子筛选

- 每个训练折内部按 Rank IC 对候选因子打分。
- 排除未来字段、目标标签字段和高缺失字段。
- 默认缺失率上限 `0.35`，最小 `abs_mean_ic >= 0.005`。
- 策略定义中使用 Top120 候选因子筛选流程，最终投产归档有效因子为 83 个。
- 因子列表见 `factors.json`。

## 买入规则

- 信号生成：T 日收盘后使用 T 日预测结果生成信号。
- 执行日期：T+1 交易日。
- 买入时间：`09:31:00`。
- 买入条件：
- `pred_prob` 进入当日全市场 q95。
- 成交额 `amount >= 800000`。
- 换手率 `turnover_rate >= 2.0`。
- 总市值 `total_mv <= 1000000`，按项目数据口径解释。
- 排除 ST、退市股、当前涨停/不可交易股。
- 在剩余候选中按 `pred_prob` 降序选择 Top1。
- 最大持仓数：1。
- 目标仓位：`0.98`。

## 卖出规则

- 每日先卖后买，卖出时间 `09:30:00`，买入时间 `09:31:00`。
- 计划卖出：持仓满 5 个交易日后卖出。
- 评分退出：持仓满 2 个交易日后，如果当前决策日评分 `current_score <= entry_score * 0.95`，提前卖出。
- 不延长持仓：V1.0 中 `max_holding_days = holding_days = 5`。
- 可交易性检查：开盘涨停不买，开盘跌停不卖；买卖均按开盘价限价单执行。

## 掘金复现口径

- Python：`C:/Users/wangj/.conda/envs/my_quant/python.exe`。
- 策略目录：`D:/work/dfcf/juejin/strategy/ce327750-634e-11f1-b8d7-10ffe0295517`。
- 信号文件：`D:/work/quant/quant_mcp/quant/data_file/gm_signals_prod_aligned_4y_liq_enh_top1_q95_amt8e5_turn2_mv1e6_clean_nost_delist.csv`。
- 评分库：`D:/work/quant/quant_mcp/quant/data_file/odb.db`。
- 评分表：`stock_predict_data_executable_5d_open_return_prod_aligned_202206_202606`。
- 回测窗口：`2024-06-05 09:00:00` 到 `2026-07-10 15:30:00`。
- 初始资金：`600000`。
- 滑点：`0.0015`。
- 复现配置见 `reproduction_v1_0.json`。

## V1.0 回测效果

当前掘金环境 V1.0 复测结果：

| 指标 | 数值 |
| --- | ---: |
| 累计收益 | 5009.93% |
| 年化收益 | 2387.24% |
| 最大回撤 | 27.34% |
| 夏普比率 | 1.55 |
| Calmar | 87.31 |
| 开仓 / 平仓 | 94 / 93 |
| 胜率 | 61.29% |

证据日志：

- `data_file/reports/prod_liq_prime_one_juejin_verify_20260615/prod_liq_prime_one_h5_score095_mh2_sell0930_buy0931_window20240605.log`

## 历史归档结果

2026-06-13 原始归档掘金回测结果保留为历史证据：累计收益约 1152.09%，年化收益约 385.08%，最大回撤约 28.64%，夏普比率约 1.31，开仓 102 次、平仓 101 次，胜率约 61.39%。

该结果与 V1.0 当前环境复测结果不同，因此 V1.0 复现和测量应以 `reproduction_v1_0.json` 和 `validation.json` 中的 `production_v1_0_gm_backtest` 为准。

## 注意事项

- 掘金策略目录不在当前 Git 仓库中，正式复现时必须同时保存 `main.py` 快照。
- 信号 CSV 中的 `holding_days` 字段必须保持为 `5`，否则会覆盖命令行持有期。
- 本策略为历史回测和投产规则归档，不构成未来收益承诺。
