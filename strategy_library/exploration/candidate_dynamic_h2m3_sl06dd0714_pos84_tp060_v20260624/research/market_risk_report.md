# pos84_dd0714 市场风险覆盖验证

## 当前结论

本轮固定当前高收益候选，只测试粗粒度指数/市场宽度风险覆盖。该规则使用信号日市场行情，不改变股票池，不使用月份、行业或未来状态筛历史样本。

| 候选 | 指数风控 | 宽度风控 | 全周期年化 | Sharpe | 最大回撤 | recent60 年化 | recent60 Sharpe | 近期开仓最差年化 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base_no_market_risk | False | False | 667.36% | 1.96 | 39.19% | 33.25% | 0.68 | 165.53% |
| index_cc3_intra3_scale60 | True | False | 577.84% | 1.93 | 37.02% | 35.30% | 0.72 | 178.40% |
| index_cc4_intra4_scale50 | True | False | 569.52% | 1.95 | 38.48% | 28.91% | 0.60 | 172.44% |
| breadth_up08_avgm3_scale50 | False | True | 530.05% | 1.88 | 36.71% | 28.91% | 0.60 | 165.27% |
| breadth_up12_avgm25_scale60 | False | True | 473.69% | 1.82 | 37.02% | 36.79% | 0.70 | 157.79% |
| combo_index4_breadth08_scale50 | True | True | 454.25% | 1.82 | 38.78% | 27.36% | 0.57 | 152.61% |

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所：`0`
- ST / 风险警示：`0`
- 退市：`0`
- 开盘涨停：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos84_dd0714_market_risk_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos84_dd0714_market_risk_20260624\summary.csv`
- 硬过滤审计：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos84_dd0714_market_risk_20260624\hard_gate_audit.json`
- 信号文件：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos84_dd0714_market_risk_20260624\signals`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos84_dd0714_market_risk_20260624\logs`