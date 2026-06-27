# pos76 日内止损邻域实验

## 当前结论

本实验固定入场信号、目标仓位 76%、动态持有和 dd_06_12 账户回撤缩放，只调整日内止损阈值。目标是检查 `6%` 止损是否为局部偶然点。

| 候选 | 日内止损 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pos76_sl060_dd0612 | 6.0% | 395.64% | 2.02 | 37.01% | 44.79% | 1.01 | 19.75% | 161.88% |
| pos76_sl065_dd0612 | 6.5% | 353.16% | 1.97 | 37.01% | 44.79% | 1.01 | 19.75% | 161.88% |
| pos76_sl070_dd0612 | 7.0% | 353.16% | 1.97 | 37.01% | 44.79% | 1.01 | 19.75% | 161.88% |
| pos76_sl050_dd0612 | 5.0% | 310.38% | 1.96 | 37.01% | 45.36% | 1.03 | 19.75% | 162.07% |
| pos76_sl055_dd0612 | 5.5% | 310.23% | 1.96 | 37.01% | 44.79% | 1.01 | 19.75% | 161.88% |

## 硬过滤审计

- 审计文件数：`5`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- ST / 风险警示：`0`
- 退市：`0`
- 开盘涨停：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_stoploss_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_stoploss_20260624\summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_stoploss_20260624\hard_gate_audit.json`
- 信号文件：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_stoploss_20260624\signals`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_stoploss_20260624\logs`