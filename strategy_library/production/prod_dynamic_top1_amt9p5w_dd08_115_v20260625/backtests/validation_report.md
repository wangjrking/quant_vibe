# pos8975_scale7555 完整验证报告

## 当前结论

- 验证对象：`dynamic_h2_m3_c0975_w78_5d12_3d10_pos8975_e097_mh1_sl06_dd0712_tp070_scale7555`。
- 该版本只在 c975_tp070 基础上微调目标仓位和账户回撤缩放：目标仓位 `89.75%`，soft/hard 缩放 `75%/55%`。
- 输入仍为当前 formal L4 多周期模型；不使用行业、月份、日期排除，不使用最新状态回填历史样本，不使用 1D 模型。
- 风格暴露和风格漂移仅作为提示项，不作为单独强准入。
- 全周期年化：`745.91%`；Sharpe：`1.97`；最大回撤：`39.25%`。
- 近期开仓最低年化：`157.11%`。

是否能替代当前候选，需同时看下方低路径依赖、贡献压力和硬过滤审计；本报告不修改生产 registry，不生成正式交易信号。

## 时间切片

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 745.91% | 1.97 | 39.25% | 201 | 40.39% |
| slice_2024h2 | 365.72% | 1.42 | 34.02% | 55 | 40.61% |
| slice_2025h1 | 126.67% | 1.58 | 29.37% | 40 | 38.48% |
| slice_2025h2 | 216.92% | 2.76 | 15.53% | 57 | 43.93% |
| slice_2026ytd | 170.76% | 2.39 | 21.70% | 49 | 44.71% |
| slice_recent120 | 238.75% | 2.80 | 21.69% | 52 | 44.41% |
| slice_recent60 | 46.82% | 0.92 | 21.66% | 22 | 46.41% |

## 低路径依赖

| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20250701 | 7 | 327.35% | 329.75% | 348.02% | 21.71% |
| 20251009 | 7 | 163.01% | 168.74% | 190.43% | 21.71% |
| 20260105 | 7 | 157.11% | 198.92% | 243.91% | 21.71% |

## 贡献压力

- 最高代理贡献股票：`300573.SZ`，占正代理贡献 `11.35%`。
- 最高代理贡献日期：`20240925`，占正代理贡献 `4.43%`。
- 最高代理贡献月份：`202409`，占正代理贡献 `12.04%`。

| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | 原始信号 | 745.91% | 1.97 | 39.25% | 170.76% |
| drop_top_1pct | 去最高 1% 代理信号 5 条 | 649.61% | 1.92 | 39.25% | 170.76% |
| drop_top_day | 去最高代理信号日 20240925 | 745.91% | 1.97 | 39.25% | 170.76% |
| drop_top_month | 去最高代理月份 202409 | 696.09% | 1.96 | 39.26% | 170.76% |
| drop_top_stock | 去最高代理股票 300573.SZ | 742.97% | 1.97 | 39.26% | 170.76% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所命中：`0`
- 买入日 ST / 风险警示：`0`
- 买入日退市：`0`
- 买入日开盘涨停：`0`

## 证据路径

- 时间切片：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos8975_scale7555_validation_20260625\time_slices.csv`
- 低路径依赖：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos8975_scale7555_validation_20260625\nearby_summary.csv`
- 贡献压力：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos8975_scale7555_validation_20260625\stress_summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos8975_scale7555_validation_20260625\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos8975_scale7555_validation_20260625\logs`
