# pos84_dd0714 止盈邻域验证

## 当前结论

本轮固定 Top1 信号、目标仓位 84%、6% 止损、`dd0714_s8060` 账户回撤缩放和动态持有规则，只调整止盈阈值。该验证用于检查当前高收益候选的 6% 止盈是否稳健，以及是否存在更好的近期持续性折中点。

| 候选 | 止盈阈值 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pos84_dd0714_tp060 | 6.0% | 667.36% | 1.96 | 39.19% | 33.25% | 0.68 | 22.56% | 165.53% |
| pos84_dd0714_tp065 | 6.5% | 626.36% | 1.91 | 39.19% | 33.25% | 0.68 | 22.56% | 165.53% |
| pos84_dd0714_tp080 | 8.0% | 611.86% | 1.95 | 40.12% | 35.59% | 0.73 | 22.55% | 167.54% |
| pos84_dd0714_tp055 | 5.5% | 586.51% | 1.94 | 39.19% | 33.25% | 0.68 | 22.56% | 165.53% |
| pos84_dd0714_tp070 | 7.0% | 567.66% | 1.88 | 39.19% | 33.25% | 0.68 | 22.56% | 165.53% |
| pos84_dd0714_tp050 | 5.0% | 565.78% | 1.91 | 39.18% | 33.25% | 0.68 | 22.56% | 165.53% |
| pos84_dd0714_tpnone | 无 | 500.32% | 1.95 | 40.11% | 42.86% | 0.86 | 22.61% | 174.47% |
| pos84_dd0714_tp040 | 4.0% | 493.08% | 1.81 | 39.18% | -5.23% | -0.13 | 22.59% | 132.89% |
| pos84_dd0714_tp120 | 12.0% | 470.83% | 1.91 | 40.12% | 35.59% | 0.73 | 22.55% | 167.54% |
| pos84_dd0714_tp100 | 10.0% | 465.71% | 1.91 | 40.11% | 35.59% | 0.73 | 22.55% | 167.54% |

## 硬过滤审计

- 审计文件数：`10`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- ST / 风险警示：`0`
- 退市：`0`
- 开盘涨停：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos84_dd0714_takeprofit_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos84_dd0714_takeprofit_20260624\summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos84_dd0714_takeprofit_20260624\hard_gate_audit.json`
- 信号文件：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos84_dd0714_takeprofit_20260624\signals`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos84_dd0714_takeprofit_20260624\logs`