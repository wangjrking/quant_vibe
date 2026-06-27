# sl06_dd0611_pos90_tp060 完整验证

## 当前结论

验证对象：`dynamic_h2_m3_c098_w78_5d12_3d10_pos90_e097_mh1_sl06_dd0611_tp060`。该版本固定 formal L4 多周期模型入场、Top1 单票、目标仓位 `90%`、动态持有 `h2_m3_c098`、分数退出 `0.97`、日内 `6%` 止损和通用 `6%` 止盈；账户回撤缩放为 `soft=6% / hard=11% / recover=3% / scale=0.70/0.50`。

本报告用于判断 `pos90` 是否能作为比 `pos86` 更强的收益增强候选。收益指标只作为弱准入排序；持续性、低路径依赖、贡献压力、硬过滤和掘金复现是强准入依据。

## 时间切片

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 593.25% | 1.91 | 39.06% | 203 | 37.64% |
| slice_2024h2 | 372.28% | 1.50 | 34.31% | 55 | 39.40% |
| slice_2025h1 | 109.54% | 1.53 | 29.30% | 40 | 34.16% |
| slice_2025h2 | 194.63% | 2.52 | 13.76% | 59 | 41.75% |
| slice_2026ytd | 169.76% | 2.41 | 20.84% | 49 | 43.90% |
| slice_recent120 | 237.15% | 2.83 | 20.89% | 52 | 43.85% |
| slice_recent60 | 51.14% | 1.02 | 20.79% | 22 | 45.32% |

## 低路径依赖

| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20250701 | 7 | 300.94% | 302.34% | 324.12% | 20.90% |
| 20251009 | 7 | 147.83% | 164.82% | 186.75% | 20.90% |
| 20260105 | 7 | 156.57% | 197.44% | 242.06% | 20.90% |

## 贡献集中压力

- 最高代理贡献股票：`300573.SZ`，占正代理贡献 `11.35%`。
- 最高代理贡献日：`20240925`，占正代理贡献 `4.43%`。
- 最高代理贡献月份：`202409`，占正代理贡献 `12.04%`。

| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | 原始信号 | 593.25% | 1.91 | 39.06% | 169.76% |
| drop_top_1pct | 去最高 1% 代理信号 5 条 | 528.40% | 1.86 | 39.06% | 169.76% |
| drop_top_day | 去最高代理信号日 20240925 | 593.25% | 1.91 | 39.06% | 169.76% |
| drop_top_month | 去最高代理月份 202409 | 554.96% | 1.89 | 39.05% | 169.76% |
| drop_top_stock | 去最高代理股票 300573.SZ | 590.98% | 1.90 | 39.06% | 169.76% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所命中：`0`
- 买入日 ST / 风险警示：`0`
- 买入日退市：`0`
- 买入日开盘涨停：`0`

## 证据路径

- 时间切片：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos90_tp060_validation_20260624\time_slices.csv`
- 低路径依赖：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos90_tp060_validation_20260624\nearby_summary.csv`
- 贡献压力：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos90_tp060_validation_20260624\stress_summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos90_tp060_validation_20260624\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos90_tp060_validation_20260624\logs`
