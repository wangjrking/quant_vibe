# pos76 账户回撤缩放邻域验证

## 当前结论

本轮固定入场信号、目标仓位 76%、6% 日内止损、动态持有规则，只调整账户回撤缩放参数。该验证用于判断收益改善是否来自回撤缩放邻域，而不是来自选股规则变化。

| 候选 | 风控 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pos76_sl06_no_dd | 关闭 | 476.92% | 1.92 | 49.84% | 78.59% | 1.52 | 24.74% | 163.26% |
| pos76_sl06_dd0612_s8060 | 6%/12%, scale 80%/60% | 395.64% | 2.02 | 37.01% | 44.79% | 1.01 | 19.75% | 161.88% |
| pos76_sl06_dd0814_s8565 | 8%/14%, scale 85%/65% | 392.25% | 1.98 | 40.42% | 36.72% | 0.81 | 21.32% | 161.32% |
| pos76_sl06_dd0714_s8060 | 7%/14%, scale 80%/60% | 378.90% | 1.98 | 38.59% | 40.15% | 0.91 | 20.63% | 158.41% |
| pos76_sl06_dd0612_s8565 | 6%/12%, scale 85%/65% | 378.22% | 1.98 | 38.88% | 41.10% | 0.91 | 20.60% | 159.72% |
| pos76_sl06_dd0510_s8060 | 5%/10%, scale 80%/60% | 326.56% | 1.94 | 37.45% | 47.19% | 1.08 | 19.29% | 162.28% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- ST / 风险警示：`0`
- 退市：`0`
- 开盘涨停：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_dd_neighborhood_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_dd_neighborhood_20260624\summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_dd_neighborhood_20260624\hard_gate_audit.json`
- 信号文件：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_dd_neighborhood_20260624\signals`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_dd_neighborhood_20260624\logs`