# pos89 非仓位参数粗邻域验证

## 当前结论

本轮固定 formal L4 输入、Top1 股票池、ST/BJ/退市/涨停硬过滤和目标仓位 89%，只调整卖出、持有、止损止盈和账户回撤缩放参数。收益指标只作为弱排序，强准入优先看最大回撤、recent60、近期开仓和硬过滤。

| 候选 | 年化 | Sharpe | 最大回撤 | recent60 年化 | 近期开仓最差年化 | 参数摘要 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `continue_c975` | 661.98% | 1.97 | 36.91% | 50.72% | 167.29% | e=0.970, c=0.975, mh=3, sl=6.00%, tp=6.00%, dd=7%/12% |
| `take_tp07` | 657.07% | 1.95 | 36.91% | 50.72% | 167.29% | e=0.970, c=0.980, mh=3, sl=6.00%, tp=7.00%, dd=7%/12% |
| `continue_c985` | 638.00% | 1.98 | 37.39% | 50.72% | 198.90% | e=0.970, c=0.985, mh=3, sl=6.00%, tp=6.00%, dd=7%/12% |
| `dd0813` | 628.99% | 1.92 | 36.91% | 47.85% | 167.58% | e=0.970, c=0.980, mh=3, sl=6.00%, tp=6.00%, dd=8%/13% |
| `stop_sl05` | 621.37% | 1.93 | 36.91% | 50.90% | 167.44% | e=0.970, c=0.980, mh=3, sl=5.00%, tp=6.00%, dd=7%/12% |
| `base_pos89_e970_c980_mh3_sl06_tp06_dd0712` | 621.06% | 1.93 | 36.91% | 50.72% | 167.29% | e=0.970, c=0.980, mh=3, sl=6.00%, tp=6.00%, dd=7%/12% |
| `exit_e965` | 621.06% | 1.93 | 36.91% | 50.72% | 167.29% | e=0.965, c=0.980, mh=3, sl=6.00%, tp=6.00%, dd=7%/12% |
| `stop_sl07` | 609.88% | 1.92 | 36.91% | 50.72% | 167.29% | e=0.970, c=0.980, mh=3, sl=7.00%, tp=6.00%, dd=7%/12% |
| `dd0611` | 575.98% | 1.91 | 38.70% | 50.72% | 167.86% | e=0.970, c=0.980, mh=3, sl=6.00%, tp=6.00%, dd=6%/11% |
| `exit_e975` | 559.24% | 1.91 | 36.91% | 50.72% | 167.29% | e=0.975, c=0.980, mh=3, sl=6.00%, tp=6.00%, dd=7%/12% |
| `take_tp05` | 500.99% | 1.86 | 36.91% | 50.72% | 167.29% | e=0.970, c=0.980, mh=3, sl=6.00%, tp=5.00%, dd=7%/12% |
| `hold_mh4` | 452.41% | 1.56 | 35.02% | -55.09% | 7.51% | e=0.970, c=0.980, mh=4, sl=6.00%, tp=6.00%, dd=7%/12% |
| `hold_mh2` | 150.89% | 0.83 | 44.47% | -91.12% | 16.09% | e=0.970, c=0.980, mh=2, sl=6.00%, tp=6.00%, dd=7%/12% |

## 硬过滤审计

- 审计文件数：`13`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所命中：`0`
- ST / 风险警示命中：`0`
- 退市命中：`0`
- 开盘涨停买入命中：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624\summary.csv`
- 硬过滤：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624\hard_gate_audit.json`
- 日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624\logs`
