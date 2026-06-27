# c975 窄邻域参数验证

## 当前结论

本轮固定 formal L4 输入、Top1 股票池、ST/BJ/退市/涨停硬过滤和非行业非月份口径，只微调连续持仓保留阈值、止盈、止损、目标仓位和账户回撤缩放。收益指标只用于排序，生产准入仍以硬过滤、低路径依赖、时间切片、贡献压力和掘金复现为准。

| 候选 | 年化 | Sharpe | 最大回撤 | recent60 年化 | 近期开仓最差年化 | 平均仓位 | 参数摘要 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `c975_tp070` | 700.07% | 1.99 | 36.91% | 50.72% | 167.29% | 39.20% | pos=89.00%, c=0.9750, sl=6.00%, tp=7.00%, dd=7%/12%, scale=70%/50% |
| `c9725_tp070` | 700.07% | 1.99 | 36.91% | 50.72% | 167.29% | 39.20% | pos=89.00%, c=0.9725, sl=6.00%, tp=7.00%, dd=7%/12%, scale=70%/50% |
| `c975_dd0813` | 670.46% | 1.96 | 36.91% | 47.85% | 167.58% | 38.92% | pos=89.00%, c=0.9750, sl=6.00%, tp=6.00%, dd=8%/13%, scale=70%/50% |
| `base_c975_pos89_tp060_sl060_dd0712` | 661.98% | 1.97 | 36.91% | 50.72% | 167.29% | 37.80% | pos=89.00%, c=0.9750, sl=6.00%, tp=6.00%, dd=7%/12%, scale=70%/50% |
| `c970` | 661.98% | 1.97 | 36.91% | 50.72% | 167.29% | 37.80% | pos=89.00%, c=0.9700, sl=6.00%, tp=6.00%, dd=7%/12%, scale=70%/50% |
| `c9725` | 661.98% | 1.97 | 36.91% | 50.72% | 167.29% | 37.80% | pos=89.00%, c=0.9725, sl=6.00%, tp=6.00%, dd=7%/12%, scale=70%/50% |
| `c975_sl055` | 661.98% | 1.97 | 36.91% | 50.72% | 167.29% | 37.80% | pos=89.00%, c=0.9750, sl=5.50%, tp=6.00%, dd=7%/12%, scale=70%/50% |
| `c9775_tp070` | 657.07% | 1.95 | 36.91% | 50.72% | 167.29% | 38.83% | pos=89.00%, c=0.9775, sl=6.00%, tp=7.00%, dd=7%/12%, scale=70%/50% |
| `c975_pos90` | 655.95% | 1.94 | 37.25% | 51.14% | 169.19% | 38.17% | pos=90.00%, c=0.9750, sl=6.00%, tp=6.00%, dd=7%/12%, scale=70%/50% |
| `c975_tp065` | 653.93% | 1.96 | 36.91% | 50.72% | 167.29% | 38.11% | pos=89.00%, c=0.9750, sl=6.00%, tp=6.50%, dd=7%/12%, scale=70%/50% |
| `c975_sl065` | 650.39% | 1.96 | 36.91% | 50.72% | 167.29% | 37.81% | pos=89.00%, c=0.9750, sl=6.50%, tp=6.00%, dd=7%/12%, scale=70%/50% |
| `c975_pos90_tp070` | 645.26% | 1.95 | 39.05% | 51.14% | 169.19% | 39.25% | pos=90.00%, c=0.9750, sl=6.00%, tp=7.00%, dd=7%/12%, scale=70%/50% |
| `c975_pos88` | 637.67% | 1.96 | 36.57% | 46.87% | 162.91% | 37.45% | pos=88.00%, c=0.9750, sl=6.00%, tp=6.00%, dd=7%/12%, scale=70%/50% |
| `c9775` | 621.06% | 1.93 | 36.91% | 50.72% | 167.29% | 37.44% | pos=89.00%, c=0.9775, sl=6.00%, tp=6.00%, dd=7%/12%, scale=70%/50% |
| `c980` | 621.06% | 1.93 | 36.91% | 50.72% | 167.29% | 37.44% | pos=89.00%, c=0.9800, sl=6.00%, tp=6.00%, dd=7%/12%, scale=70%/50% |
| `c975_dd0611` | 614.18% | 1.96 | 38.70% | 50.72% | 167.86% | 37.60% | pos=89.00%, c=0.9750, sl=6.00%, tp=6.00%, dd=6%/11%, scale=70%/50% |
| `c975_pos91` | 613.55% | 1.91 | 39.41% | 52.12% | 171.42% | 38.48% | pos=91.00%, c=0.9750, sl=6.00%, tp=6.00%, dd=7%/12%, scale=70%/50% |
| `c975_tp075` | 546.34% | 1.93 | 37.75% | 53.26% | 169.77% | 37.51% | pos=89.00%, c=0.9750, sl=6.00%, tp=7.50%, dd=7%/12%, scale=70%/50% |
| `c975_scale060040` | 519.68% | 1.93 | 32.56% | 55.68% | 134.77% | 35.35% | pos=89.00%, c=0.9750, sl=6.00%, tp=6.00%, dd=7%/12%, scale=60%/40% |
| `c975_scale080060` | 741.05% | 1.93 | 41.04% | 42.35% | 172.19% | 40.46% | pos=89.00%, c=0.9750, sl=6.00%, tp=6.00%, dd=7%/12%, scale=80%/60% |

## 硬过滤审计

- 审计文件数：`20`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所命中：`0`
- ST / 风险警示命中：`0`
- 退市命中：`0`
- 开盘涨停买入命中：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_fine_neighborhood_20260624\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_fine_neighborhood_20260624\summary.csv`
- 准入排序：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_fine_neighborhood_20260624\summary_by_admission.csv`
- 硬过滤：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_fine_neighborhood_20260624\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_fine_neighborhood_20260624\logs`
