# sl06_dd0611_pos86_tp060 完整验证

## 当前结论

验证对象：`dynamic_h2_m3_c098_w78_5d12_3d10_pos86_e097_mh1_sl06_dd0611_tp060`。该版本固定 formal L4 多周期模型入场、Top1 单票、目标仓位 `86%`、动态持有 `h2_m3_c098`、分数退出 `0.97`、日内 `6%` 止损和通用 `6%` 止盈；账户回撤缩放为 `soft=6% / hard=11% / recover=3% / scale=0.70/0.50`。

本报告用于判断回撤中间档是否比 `dd0509_s7050` 更适合作为强准入候选。收益指标只作为弱准入排序；持续性、低路径依赖、贡献压力、硬过滤和掘金复现是强准入依据。

## 时间切片

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓数 | 平均仓位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 529.93% | 1.93 | 37.64% | 203 | 36.02% |
| slice_2024h2 | 348.51% | 1.54 | 33.02% | 55 | 37.69% |
| slice_2025h1 | 102.14% | 1.48 | 28.35% | 40 | 33.89% |
| slice_2025h2 | 183.73% | 2.53 | 13.18% | 59 | 39.91% |
| slice_2026ytd | 161.38% | 2.44 | 19.97% | 49 | 42.13% |
| slice_recent120 | 224.64% | 2.88 | 19.99% | 52 | 41.91% |
| slice_recent60 | 49.59% | 1.04 | 19.92% | 22 | 43.26% |

## 低路径依赖

| 锚点 | 起点数 | 年化最小 | 年化中位 | 年化最大 | 最大回撤最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20250701 | 7 | 279.12% | 280.88% | 297.13% | 20.05% |
| 20251009 | 7 | 140.27% | 155.79% | 176.30% | 20.05% |
| 20260105 | 7 | 121.54% | 186.79% | 229.63% | 20.05% |

## 贡献集中压力

- 最高代理贡献股票：`300573.SZ`，占正代理贡献 `11.35%`。
- 最高代理贡献日：`20240925`，占正代理贡献 `4.43%`。
- 最高代理贡献月份：`202409`，占正代理贡献 `12.04%`。

| 压力变体 | 说明 | 年化 | Sharpe | 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | 原始信号 | 529.93% | 1.93 | 37.64% | 161.38% |
| drop_top_1pct | 去最高 1% 代理信号 5 条 | 473.14% | 1.88 | 37.64% | 161.38% |
| drop_top_day | 去最高代理信号日 20240925 | 529.93% | 1.93 | 37.64% | 161.38% |
| drop_top_month | 去最高代理月份 202409 | 495.72% | 1.92 | 37.64% | 161.38% |
| drop_top_stock | 去最高代理股票 300573.SZ | 527.91% | 1.93 | 37.64% | 161.38% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所命中：`0`
- 买入日 ST / 风险警示：`0`
- 买入日退市：`0`
- 买入日开盘涨停：`0`

## 证据路径

- 时间切片：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos86_tp060_validation_20260624\time_slices.csv`
- 低路径依赖：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos86_tp060_validation_20260624\nearby_summary.csv`
- 贡献压力：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos86_tp060_validation_20260624\stress_summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos86_tp060_validation_20260624\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos86_tp060_validation_20260624\logs`
