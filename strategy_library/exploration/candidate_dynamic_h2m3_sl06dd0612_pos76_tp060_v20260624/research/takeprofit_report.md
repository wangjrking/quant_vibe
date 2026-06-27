# pos76 止盈邻域验证

## 当前结论

本轮固定入场信号、目标仓位 76%、6% 日内止损、`dd0612_s8060` 账户回撤缩放和动态持有规则，只调整止盈阈值。止盈是通用退出规则，不改变股票池，不使用行业、月份、日期或最新状态筛历史样本。

| 候选 | 止盈阈值 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | recent60 最大回撤 | 近期开仓最差年化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pos76_sl06_dd0612_tp060 | 6% | 503.25% | 2.03 | 36.13% | 34.95% | 0.81 | 19.81% | 154.40% |
| pos76_sl06_dd0612_tp070 | 7% | 498.94% | 2.02 | 36.14% | 34.95% | 0.81 | 19.81% | 154.40% |
| pos76_sl06_dd0612_tp080 | 8% | 478.64% | 2.03 | 37.01% | 37.18% | 0.86 | 19.80% | 156.42% |
| pos76_sl06_dd0612_tp085 | 8% | 478.64% | 2.03 | 37.01% | 37.18% | 0.86 | 19.80% | 156.42% |
| pos76_sl06_dd0612_tp075 | 8% | 477.67% | 2.03 | 37.01% | 37.18% | 0.86 | 19.80% | 156.42% |
| pos76_sl06_dd0612_tpnone | 无 | 395.64% | 2.02 | 37.01% | 44.79% | 1.01 | 19.75% | 161.88% |
| pos76_sl06_dd0612_tp120 | 12% | 372.83% | 1.99 | 37.01% | 37.18% | 0.86 | 19.80% | 156.42% |
| pos76_sl06_dd0612_tp090 | 9% | 371.48% | 2.00 | 37.00% | 37.18% | 0.86 | 19.80% | 156.42% |
| pos76_sl06_dd0612_tp150 | 15% | 370.67% | 2.00 | 37.02% | 44.79% | 1.01 | 19.75% | 161.88% |
| pos76_sl06_dd0612_tp200 | 20% | 370.67% | 2.00 | 37.02% | 44.79% | 1.01 | 19.75% | 161.88% |
| pos76_sl06_dd0612_tp100 | 10% | 369.63% | 1.99 | 37.01% | 37.18% | 0.86 | 19.80% | 156.42% |

## 硬过滤审计

- 审计文件数：`11`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- ST / 风险警示：`0`
- 退市：`0`
- 开盘涨停：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_takeprofit_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_takeprofit_20260624\summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_takeprofit_20260624\hard_gate_audit.json`
- 信号文件：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_takeprofit_20260624\signals`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_takeprofit_20260624\logs`