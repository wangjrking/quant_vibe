# no_dd 仓位邻域验证

## 当前结论

本轮固定同一套 Top1 信号、动态持有和 6% 日内止损，关闭账户回撤缩放，只调整目标仓位。目的不是追求单一最高收益，而是检查 no_dd 的高收益能否通过降低仓位压低回撤，并保持启动点持续性。

| 候选 | 目标仓位 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_dd_pos76_sl06 | 76% | 476.92% | 1.92 | 49.84% | 78.59% | 1.52 | 24.74% | 163.26% |
| no_dd_pos74_sl06 | 74% | 448.46% | 1.94 | 48.87% | 78.02% | 1.55 | 24.13% | 159.38% |
| no_dd_pos72_sl06 | 72% | 421.41% | 1.95 | 47.87% | 77.29% | 1.59 | 23.52% | 154.84% |
| no_dd_pos70_sl06 | 70% | 396.99% | 1.97 | 46.87% | 75.25% | 1.59 | 23.05% | 151.28% |
| no_dd_pos66_sl06 | 66% | 350.55% | 2.00 | 44.83% | 73.88% | 1.66 | 21.80% | 134.67% |
| no_dd_pos62_sl06 | 62% | 306.14% | 2.03 | 42.71% | 19.82% | 0.49 | 20.54% | 125.98% |
| no_dd_pos58_sl06 | 58% | 267.42% | 2.05 | 40.52% | 18.07% | 0.48 | 19.41% | 118.04% |

## 硬过滤审计

- 审计文件数：`7`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- ST / 风险警示：`0`
- 退市：`0`
- 开盘涨停：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_no_dd_position_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_no_dd_position_20260624\summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_no_dd_position_20260624\hard_gate_audit.json`
- 信号文件：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_no_dd_position_20260624\signals`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_no_dd_position_20260624\logs`