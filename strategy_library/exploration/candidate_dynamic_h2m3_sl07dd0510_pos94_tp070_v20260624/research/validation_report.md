# sl07_dd0510_pos94_tp070 完整验证

## 当前结论

验证对象：`dynamic_h2_m3_c098_w78_5d12_3d10_pos94_e097_mh1_sl07_dd0510_tp070`。该版本固定 formal L4 多周期模型入场、Top1 单票、目标仓位 `94%`、动态持有 `h2_m3_c098`、分数退出 `0.97`、日内 `7%` 止损、账户回撤缩放 `soft=5% / hard=10% / recover=3% / scale=0.70/0.50`，新增通用 `7%` 止盈。

止盈属于通用退出规则，不改变股票池，不使用行业、月份、日期或最新状态筛历史样本。本报告用于判断收益增强是否同时满足强准入里的持续性、低路径依赖和压力验证要求。

## 时间切片

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 514.46% | 1.82 | 39.66% | 203 | 38.75% |
| slice_2024h2 | 308.93% | 1.33 | 34.74% | 55 | 40.16% |
| slice_2025h1 | 129.28% | 1.61 | 30.28% | 40 | 34.65% |
| slice_2025h2 | 184.63% | 2.35 | 14.36% | 59 | 42.56% |
| slice_2026ytd | 172.70% | 2.39 | 20.30% | 49 | 45.00% |
| slice_recent120 | 244.07% | 2.82 | 20.31% | 52 | 44.97% |
| slice_recent60 | 52.78% | 1.04 | 20.30% | 22 | 46.37% |

## 低路径依赖

| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20250701 | 7 | 295.57% | 297.49% | 319.31% | 20.39% |
| 20251009 | 7 | 152.45% | 169.70% | 199.19% | 20.37% |
| 20260105 | 7 | 154.56% | 201.90% | 250.00% | 20.34% |

## 贡献集中压力

- 最高代理贡献股票：`300573.SZ`，占正代理贡献 `11.35%`。
- 最高代理贡献日：`20240925`，占正代理贡献 `4.43%`。
- 最高代理贡献月份：`202409`，占正代理贡献 `12.04%`。

| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | 原始信号 | 514.46% | 1.82 | 39.66% | 172.70% |
| drop_top_1pct | 去最高 1% 代理信号 5 条 | 453.41% | 1.76 | 39.66% | 172.70% |
| drop_top_day | 去最高代理信号日 20240925 | 514.46% | 1.82 | 39.66% | 172.70% |
| drop_top_month | 去最高代理月份 202409 | 484.54% | 1.81 | 39.66% | 172.70% |
| drop_top_stock | 去最高代理股票 300573.SZ | 512.44% | 1.82 | 39.66% | 172.70% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- 买入日 ST / 风险警示：`0`
- 买入日退市：`0`
- 买入日开盘涨停：`0`

## 证据路径

- 时间切片：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl07_dd0510_pos94_tp070_validation_20260624\time_slices.csv`
- 低路径依赖：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl07_dd0510_pos94_tp070_validation_20260624\nearby_summary.csv`
- 贡献压力：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl07_dd0510_pos94_tp070_validation_20260624\stress_summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl07_dd0510_pos94_tp070_validation_20260624\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl07_dd0510_pos94_tp070_validation_20260624\logs`