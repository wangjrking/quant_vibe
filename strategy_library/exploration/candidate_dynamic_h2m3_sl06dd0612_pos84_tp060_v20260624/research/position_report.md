# tp060 仓位邻域验证

## 当前结论

本轮固定同一套 Top1 信号、动态持有、6% 止盈、6% 止损和 `dd0612_s8060` 账户回撤缩放，只调整目标仓位。该验证用于判断收益增强候选是否还能通过风险暴露参数提高年化，并保持强准入指标。

| 候选 | 目标仓位 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp060_pos84 | 84% | 586.56% | 1.93 | 39.70% | 37.86% | 0.78 | 21.70% | 159.33% |
| tp060_pos82 | 82% | 548.74% | 1.94 | 38.93% | 37.58% | 0.79 | 21.15% | 155.27% |
| tp060_pos80 | 80% | 509.88% | 1.95 | 38.16% | 36.29% | 0.79 | 20.75% | 151.47% |
| tp060_pos76 | 76% | 503.25% | 2.03 | 36.13% | 34.95% | 0.81 | 19.81% | 154.40% |
| tp060_pos78 | 78% | 496.19% | 1.99 | 37.37% | 35.71% | 0.80 | 20.20% | 158.54% |
| tp060_pos74 | 74% | 488.84% | 2.06 | 35.35% | 34.95% | 0.83 | 19.26% | 149.97% |
| tp060_pos72 | 72% | 446.58% | 2.04 | 34.56% | 29.64% | 0.72 | 19.61% | 113.08% |
| tp060_pos68 | 68% | 398.75% | 2.08 | 32.95% | 27.73% | 0.72 | 18.64% | 106.49% |

## 硬过滤审计

- 审计文件数：`8`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- ST / 风险警示：`0`
- 退市：`0`
- 开盘涨停：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_tp060_position_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_tp060_position_20260624\summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_tp060_position_20260624\hard_gate_audit.json`
- 信号文件：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_tp060_position_20260624\signals`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_tp060_position_20260624\logs`