# sl06_dd0712_pos89_tp060 完整验证

## 当前结论

验证对象：`dynamic_h2_m3_c098_w78_5d12_3d10_pos89_e097_mh1_sl06_dd0712_tp060`。该版本只在已验证的 `dd0712_tp060` 规则上把目标仓位从 `90%` 调整为 `89%`，其余选股、持有、止损止盈、账户回撤缩放和硬过滤口径保持一致。

收益指标只作为弱准入排序；是否能替代现有候选，仍以持续性、低路径依赖、贡献压力、硬过滤和掘金复现证据为准。

## 时间切片

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 621.06% | 1.93 | 36.91% | 203 | 37.44% |
| slice_2024h2 | 408.76% | 1.59 | 32.06% | 55 | 39.62% |
| slice_2025h1 | 104.61% | 1.44 | 29.20% | 40 | 35.89% |
| slice_2025h2 | 188.80% | 2.50 | 15.40% | 59 | 41.20% |
| slice_2026ytd | 167.29% | 2.41 | 20.67% | 49 | 43.62% |
| slice_recent120 | 233.29% | 2.84 | 20.65% | 52 | 43.52% |
| slice_recent60 | 50.72% | 1.03 | 20.60% | 22 | 44.82% |

## 低路径依赖

| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20250701 | 7 | 280.14% | 293.45% | 310.07% | 20.69% |
| 20251009 | 7 | 143.91% | 160.52% | 182.47% | 20.69% |
| 20260105 | 7 | 153.78% | 194.88% | 238.37% | 20.68% |

## 贡献集中压力

- 最高代理贡献股票：`300573.SZ`，占正代理贡献 `11.35%`。
- 最高代理贡献日：`20240925`，占正代理贡献 `4.43%`。
- 最高代理贡献月份：`202409`，占正代理贡献 `12.04%`。

| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | 原始信号 | 621.06% | 1.93 | 36.91% | 167.29% |
| drop_top_1pct | 去最高 1% 代理信号 5 条 | 552.05% | 1.87 | 36.91% | 167.29% |
| drop_top_day | 去最高代理信号日 20240925 | 621.06% | 1.93 | 36.91% | 167.29% |
| drop_top_month | 去最高代理月份 202409 | 578.68% | 1.91 | 36.91% | 167.29% |
| drop_top_stock | 去最高代理股票 300573.SZ | 618.42% | 1.92 | 36.91% | 167.29% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所命中：`0`
- 买入日 ST / 风险警示：`0`
- 买入日退市：`0`
- 买入日开盘涨停：`0`

## 证据路径

- 时间切片：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_tp060_validation_20260624\time_slices.csv`
- 低路径依赖：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_tp060_validation_20260624\nearby_summary.csv`
- 贡献压力：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_tp060_validation_20260624\stress_summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_tp060_validation_20260624\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_tp060_validation_20260624\logs`
