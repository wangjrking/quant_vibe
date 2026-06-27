# dynamic h2_m3_c098 + sl06_dd0612 + pos76 完整验证

## 当前结论

验证对象：`dynamic_h2_m3_c098_w78_5d12_3d10_pos76_e097_mh1_sl06_dd0612`。该版本在 dynamic h2_m3_c098 基础上增加日内 `6%` 止损、账户回撤缩放 `soft=6% / hard=12% / scale=0.80/0.60`，并将单票目标仓位提高到 `76%`。

该规则不改变入场信号，只验证可实时执行的退出风控是否改善回撤、recent60 和低路径依赖。

## 时间切片

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 395.64% | 2.02 | 37.01% | 202 | 35.10% |
| slice_2024h2 | 203.30% | 1.35 | 31.08% | 54 | 35.09% |
| slice_2025h1 | 89.73% | 1.44 | 27.24% | 40 | 35.38% |
| slice_2025h2 | 158.60% | 2.52 | 13.23% | 59 | 36.90% |
| slice_2026ytd | 161.88% | 2.66 | 19.85% | 49 | 39.14% |
| slice_recent120 | 214.12% | 3.05 | 19.79% | 52 | 38.95% |
| slice_recent60 | 44.79% | 1.01 | 19.75% | 22 | 40.73% |

## 低路径依赖

| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20250701 | 7 | 251.77% | 254.13% | 266.93% | 19.87% |
| 20251009 | 7 | 134.20% | 148.15% | 165.31% | 19.86% |
| 20260105 | 7 | 150.55% | 185.17% | 219.13% | 19.85% |

## 贡献集中压力

- 最高代理股票：`300573.SZ`，占正代理贡献 `11.35%`。
- 最高代理日：`20240925`，占正代理贡献 `4.43%`。
- 最高代理月份：`202409`，占正代理贡献 `12.04%`。

| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | 原始信号 | 395.64% | 2.02 | 37.01% | 161.88% |
| drop_top_1pct | 去最高 1% 代理信号 5 条 | 210.00% | 1.60 | 41.30% | 161.88% |
| drop_top_day | 去最高代理信号日 20240925 | 395.64% | 2.02 | 37.01% | 161.88% |
| drop_top_month | 去最高代理月份 202409 | 397.62% | 2.03 | 37.02% | 161.88% |
| drop_top_stock | 去最高代理股票 300573.SZ | 402.75% | 2.03 | 37.01% | 161.88% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- 买入日 ST / 风险警示：`0`
- 买入日退市：`0`
- 买入日开盘涨停：`0`

## 证据路径

- 时间切片：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos76_validation_20260624\time_slices.csv`
- 低路径依赖：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos76_validation_20260624\nearby_summary.csv`
- 贡献压力：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos76_validation_20260624\stress_summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos76_validation_20260624\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos76_validation_20260624\logs`