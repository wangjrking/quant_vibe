# sl06_dd0612 仓位邻域实验

## 当前结论

本实验固定入场信号、动态持有、日内 6% 止损和 dd_06_12 账户回撤缩放，只调整信号内 `target_pct`。目标是检查是否能在不破坏强准入的情况下提升收益。

| 候选 | 目标仓位 | 全周期年化 | Sharpe | 最大回撤 | 平均持仓率 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sl06_dd0612_pos74 | 74% | 373.77% | 2.04 | 36.22% | 34.30% | 43.45% | 1.01 | 19.36% | 157.54% |
| sl06_dd0612_pos80 | 80% | 361.83% | 1.90 | 39.06% | 36.21% | 46.92% | 1.00 | 20.69% | 157.98% |
| sl06_dd0612_pos68 | 68% | 327.67% | 2.09 | 33.77% | 32.14% | 36.55% | 0.94 | 18.58% | 112.59% |
| sl06_dd0612_pos62 | 62% | 267.61% | 2.13 | 32.65% | 29.47% | 34.29% | 0.97 | 17.06% | 101.42% |
| sl06_dd0612_pos56 | 56% | 221.61% | 2.16 | 30.77% | 26.80% | 26.72% | 0.83 | 16.42% | 93.60% |
| sl06_dd0612_pos50 | 50% | 178.95% | 2.14 | 27.91% | 24.34% | 22.75% | 0.78 | 15.05% | 78.49% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- ST / 风险警示：`0`
- 退市：`0`
- 开盘涨停：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_position_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_position_20260624\summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_position_20260624\hard_gate_audit.json`
- 信号文件：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_position_20260624\signals`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0612_position_20260624\logs`