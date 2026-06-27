# dynamic h2_m3_c098 + sl06_dd0612 + pos74 完整验证

## 当前结论

验证对象：`dynamic_h2_m3_c098_w78_5d12_3d10_pos74_e097_mh1_sl06_dd0612`。该版本在 dynamic h2_m3_c098 基础上增加日内 `6%` 止损、账户回撤缩放 `soft=6% / hard=12% / scale=0.80/0.60`，并将单票目标仓位提高到 `74%`。

该规则不改变入场信号，只验证可实时执行的退出风控是否改善回撤、recent60 和低路径依赖。

## 时间切片

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 373.77% | 2.04 | 36.22% | 202 | 34.30% |
| slice_2024h2 | 199.48% | 1.38 | 30.39% | 54 | 34.30% |
| slice_2025h1 | 87.51% | 1.45 | 26.60% | 40 | 34.46% |
| slice_2025h2 | 152.88% | 2.51 | 12.88% | 59 | 36.07% |
| slice_2026ytd | 157.54% | 2.68 | 19.36% | 49 | 38.13% |
| slice_recent120 | 208.84% | 3.10 | 19.34% | 52 | 37.95% |
| slice_recent60 | 43.45% | 1.01 | 19.36% | 22 | 39.73% |

## 低路径依赖

| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20250701 | 7 | 241.33% | 246.08% | 258.37% | 19.38% |
| 20251009 | 7 | 130.93% | 146.93% | 163.28% | 19.39% |
| 20260105 | 7 | 146.54% | 179.97% | 213.99% | 19.39% |

## 贡献集中压力

- 最高代理股票：`300573.SZ`，占正代理贡献 `11.35%`。
- 最高代理日：`20240925`，占正代理贡献 `4.43%`。
- 最高代理月份：`202409`，占正代理贡献 `12.04%`。

| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | 原始信号 | 373.77% | 2.04 | 36.22% | 157.54% |
| drop_top_1pct | 去最高 1% 代理信号 5 条 | 202.21% | 1.61 | 40.37% | 157.54% |
| drop_top_day | 去最高代理信号日 20240925 | 373.77% | 2.04 | 36.22% | 157.54% |
| drop_top_month | 去最高代理月份 202409 | 372.85% | 2.04 | 36.21% | 157.54% |
| drop_top_stock | 去最高代理股票 300573.SZ | 381.14% | 2.04 | 36.21% | 157.54% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- 买入日 ST / 风险警示：`0`
- 买入日退市：`0`
- 买入日开盘涨停：`0`

## 证据路径

- 时间切片：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos74_validation_20260624\time_slices.csv`
- 低路径依赖：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos74_validation_20260624\nearby_summary.csv`
- 贡献压力：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos74_validation_20260624\stress_summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos74_validation_20260624\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_pos74_validation_20260624\logs`