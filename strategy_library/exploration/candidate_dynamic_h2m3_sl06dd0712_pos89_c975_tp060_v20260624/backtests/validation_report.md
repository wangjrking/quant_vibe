# continue_c975 完整验证报告

## 当前结论

- 验证对象：`dynamic_h2_m3_c0975_w78_5d12_3d10_pos89_e097_mh1_sl06_dd0712_tp060`。
- 口径：沿用 `pos89/sl06/dd0712/tp060` 规则，只把连续持仓保留阈值从 `0.980` 调整为 `0.975`。
- 该版本仍是研究验证入口，不修改生产策略、不更新 registry、不生成正式交易信号。
- 全周期年化：`661.98%`；Sharpe：`1.97`；最大回撤：`36.91%`。
- 近期开仓最低年化：`153.78%`。

收益指标属于弱准入排序项；是否能进入生产候选仍取决于硬过滤、参数邻域、时间切片、低路径依赖、贡献压力、掘金复现和归档完整性。风格暴露和风格漂移只作为风险提示披露，不单独阻断准入。

## 时间切片

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 661.98% | 1.97 | 36.91% | 201 | 37.80% |
| slice_2024h2 | 408.76% | 1.59 | 32.06% | 55 | 39.62% |
| slice_2025h1 | 104.61% | 1.44 | 29.20% | 40 | 35.89% |
| slice_2025h2 | 212.48% | 2.76 | 15.40% | 57 | 42.70% |
| slice_2026ytd | 167.29% | 2.41 | 20.67% | 49 | 43.62% |
| slice_recent120 | 233.29% | 2.84 | 20.65% | 52 | 43.52% |
| slice_recent60 | 50.72% | 1.03 | 20.60% | 22 | 44.82% |

## 低路径依赖

| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20250701 | 7 | 302.00% | 316.56% | 338.19% | 20.69% |
| 20251009 | 7 | 160.45% | 165.50% | 182.47% | 20.69% |
| 20260105 | 7 | 153.78% | 194.88% | 238.37% | 20.68% |

## 贡献压力

- 最高代理贡献股票：`300573.SZ`，占正代理贡献 `11.35%`。
- 最高代理贡献日：`20240925`，占正代理贡献 `4.43%`。
- 最高代理贡献月份：`202409`，占正代理贡献 `12.04%`。

| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | 原始信号 | 661.98% | 1.97 | 36.91% | 167.29% |
| drop_top_1pct | 去最高 1% 代理信号 5 条 | 588.76% | 1.92 | 36.91% | 167.29% |
| drop_top_day | 去最高代理信号日 20240925 | 661.98% | 1.97 | 36.91% | 167.29% |
| drop_top_month | 去最高代理月份 202409 | 617.28% | 1.96 | 36.91% | 167.29% |
| drop_top_stock | 去最高代理股票 300573.SZ | 659.12% | 1.97 | 36.91% | 167.29% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所命中：`0`
- 买入日 ST / 风险警示：`0`
- 买入日退市：`0`
- 买入日开盘涨停：`0`

## 证据路径

- 时间切片：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_tp060_validation_20260624\time_slices.csv`
- 低路径依赖：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_tp060_validation_20260624\nearby_summary.csv`
- 贡献压力：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_tp060_validation_20260624\stress_summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_tp060_validation_20260624\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_tp060_validation_20260624\logs`
