# tp060_pos84 账户回撤缩放邻域验证

## 当前结论

本轮固定同一套 Top1 信号、目标仓位 84%、6% 止盈、6% 止损和动态持有规则，只调整账户回撤缩放参数。该验证用于判断当前收益增强候选能否在强准入风险边界内继续优化。

| 候选 | 风控 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp060_pos84_no_dd | 关闭 | 885.30% | 1.87 | 52.35% | 71.60% | 1.27 | 27.07% | 171.45% |
| tp060_pos84_dd0814_s8565 | 8%/14%, scale 85%/65% | 715.75% | 1.97 | 41.11% | 34.64% | 0.70 | 22.43% | 170.88% |
| tp060_pos84_dd0714_s8060 | 7%/14%, scale 80%/60% | 667.36% | 1.96 | 39.19% | 33.25% | 0.68 | 22.56% | 165.53% |
| tp060_pos84_dd0612_s8565 | 6%/12%, scale 85%/65% | 639.31% | 1.92 | 41.11% | 34.64% | 0.70 | 22.43% | 169.33% |
| tp060_pos84_dd0612_s8060 | 6%/12%, scale 80%/60% | 586.56% | 1.93 | 39.70% | 37.86% | 0.78 | 21.70% | 159.33% |
| tp060_pos84_dd0510_s8060 | 5%/10%, scale 80%/60% | 529.72% | 1.91 | 39.22% | 40.53% | 0.84 | 21.19% | 162.05% |
| tp060_pos84_dd0509_s7050 | 5%/9%, scale 70%/50% | 434.99% | 1.93 | 36.18% | 51.00% | 1.13 | 19.06% | 131.74% |

## 硬过滤审计

- 审计文件数：`7`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- ST / 风险警示：`0`
- 退市：`0`
- 开盘涨停：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_tp060_pos84_dd_neighborhood_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_tp060_pos84_dd_neighborhood_20260624\summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_tp060_pos84_dd_neighborhood_20260624\hard_gate_audit.json`
- 信号文件：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_tp060_pos84_dd_neighborhood_20260624\signals`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_tp060_pos84_dd_neighborhood_20260624\logs`