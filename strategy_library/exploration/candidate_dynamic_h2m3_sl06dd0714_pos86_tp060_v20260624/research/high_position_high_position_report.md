# dd0714_tp060 高仓位邻域验证

## 当前结论

本轮固定当前高收益候选的入场、动态持有、6% 止盈、6% 止损和 dd0714 账户回撤缩放，只向上测试目标仓位。目标是确认 `84%` 是否局部最优，以及更高仓位是否突破强准入风险边界。

| 候选 | 目标仓位 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dd0714_tp060_pos90 | 90% | 776.13% | 1.92 | 41.40% | 40.26% | 0.76 | 23.02% | 182.30% |
| dd0714_tp060_pos88 | 88% | 761.95% | 1.95 | 40.67% | 39.16% | 0.76 | 22.62% | 177.85% |
| dd0714_tp060_pos86 | 86% | 703.28% | 1.96 | 39.94% | 38.75% | 0.78 | 22.09% | 175.81% |
| dd0714_tp060_pos84 | 84% | 667.36% | 1.96 | 39.19% | 33.25% | 0.68 | 22.56% | 165.53% |
| dd0714_tp060_pos85 | 85% | 666.53% | 1.94 | 39.56% | 33.32% | 0.67 | 22.76% | 167.43% |
| dd0714_tp060_pos82 | 82% | 629.42% | 1.97 | 38.44% | 33.03% | 0.70 | 22.02% | 163.03% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- ST / 风险警示：`0`
- 退市：`0`
- 开盘涨停：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_dd0714_tp060_high_position_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_dd0714_tp060_high_position_20260624\summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_dd0714_tp060_high_position_20260624\hard_gate_audit.json`
- 信号文件：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_dd0714_tp060_high_position_20260624\signals`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_dd0714_tp060_high_position_20260624\logs`