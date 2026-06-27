# sl06_dd0612_pos76_tp060 完整验证

## 当前结论

验证对象：`dynamic_h2_m3_c098_w78_5d12_3d10_pos76_e097_mh1_sl06_dd0612_tp060`。该版本固定 formal L4 多周期模型入场、Top1 单票、目标仓位 `76%`、动态持有 `h2_m3_c098`、分数退出 `0.97`、日内 `6%` 止损、账户回撤缩放 `soft=6% / hard=12% / scale=0.80/0.60`，新增通用 `6%` 止盈。

止盈属于通用退出规则，不改变股票池，不使用行业、月份、日期或最新状态筛历史样本。本报告用于判断收益增强是否同时满足强准入里的持续性、低路径依赖和压力验证要求。

## 时间切片

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 503.25% | 2.03 | 36.13% | 203 | 35.08% |
| slice_2024h2 | 335.95% | 1.62 | 31.09% | 55 | 35.05% |
| slice_2025h1 | 87.30% | 1.38 | 28.67% | 40 | 34.32% |
| slice_2025h2 | 170.30% | 2.68 | 13.23% | 59 | 37.66% |
| slice_2026ytd | 154.40% | 2.56 | 19.79% | 49 | 39.14% |
| slice_recent120 | 205.13% | 2.95 | 19.84% | 52 | 38.98% |
| slice_recent60 | 34.95% | 0.81 | 19.81% | 22 | 40.81% |

## 低路径依赖

| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20250701 | 7 | 255.17% | 257.29% | 270.21% | 19.86% |
| 20251009 | 7 | 136.94% | 150.99% | 168.85% | 19.86% |
| 20260105 | 7 | 142.81% | 176.40% | 209.75% | 19.85% |

## 贡献集中压力

- 最高代理贡献股票：`300573.SZ`，占正代理贡献 `11.35%`。
- 最高代理贡献日：`20240925`，占正代理贡献 `4.43%`。
- 最高代理贡献月份：`202409`，占正代理贡献 `12.04%`。

| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | 原始信号 | 503.25% | 2.03 | 36.13% | 154.40% |
| drop_top_1pct | 去最高 1% 代理信号 5 条 | 463.39% | 1.98 | 36.13% | 154.40% |
| drop_top_day | 去最高代理信号日 20240925 | 503.25% | 2.03 | 36.13% | 154.40% |
| drop_top_month | 去最高代理月份 202409 | 474.18% | 2.02 | 36.13% | 154.40% |
| drop_top_stock | 去最高代理股票 300573.SZ | 501.66% | 2.03 | 36.14% | 154.40% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- 买入日 ST / 风险警示：`0`
- 买入日退市：`0`
- 买入日开盘涨停：`0`

## 证据路径

- 时间切片：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos76_tp060_validation_20260624\time_slices.csv`
- 低路径依赖：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos76_tp060_validation_20260624\nearby_summary.csv`
- 贡献压力：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos76_tp060_validation_20260624\stress_summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos76_tp060_validation_20260624\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos76_tp060_validation_20260624\logs`