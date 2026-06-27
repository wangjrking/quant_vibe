# 高仓位边界精调验证

## 当前结论

本轮固定 formal L4 输入、Top1 股票池、ST/BJ/退市/涨停硬过滤和非行业非月份口径，只在 `pos90/scale7555` 附近微调目标仓位与账户回撤缩放。风格暴露和风格漂移仅作为提示，不作为单独强准入。

| 候选 | 年化 | Sharpe | 最大回撤 | recent60 年化 | 近期开仓最差年化 | 平均仓位 | 参数摘要 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pos8975_scale7555` | 745.91% | 1.97 | 39.25% | 46.82% | 170.76% | 40.39% | pos=89.75%, dd=7.0%/12.0%, scale=75%/55% |
| `pos90_scale7454` | 742.53% | 1.97 | 38.94% | 47.07% | 171.06% | 40.26% | pos=90.00%, dd=7.0%/12.0%, scale=74%/54% |
| `pos895_scale7555` | 741.82% | 1.97 | 39.17% | 46.61% | 170.32% | 40.28% | pos=89.50%, dd=7.0%/12.0%, scale=75%/55% |
| `pos8975_scale7454` | 734.69% | 1.97 | 38.85% | 47.92% | 170.72% | 40.14% | pos=89.75%, dd=7.0%/12.0%, scale=74%/54% |
| `pos895_scale7353` | 731.69% | 1.98 | 38.34% | 48.63% | 169.52% | 39.88% | pos=89.50%, dd=7.0%/12.0%, scale=73%/53% |
| `pos895_scale7454` | 730.84% | 1.97 | 38.76% | 47.81% | 169.63% | 40.03% | pos=89.50%, dd=7.0%/12.0%, scale=74%/54% |
| `pos8975_scale7353` | 728.30% | 1.98 | 38.43% | 48.77% | 170.34% | 39.84% | pos=89.75%, dd=7.0%/12.0%, scale=73%/53% |
| `base_pos89_scale7050` | 700.07% | 1.99 | 36.91% | 50.72% | 167.29% | 39.20% | pos=89.00%, dd=7.0%/12.0%, scale=70%/50% |
| `pos90_scale7353` | 694.78% | 1.96 | 39.93% | 48.70% | 170.27% | 39.85% | pos=90.00%, dd=7.0%/12.0%, scale=73%/53% |
| `pos90_scale7252` | 682.59% | 1.96 | 39.64% | 49.07% | 170.22% | 39.59% | pos=90.00%, dd=7.0%/12.0%, scale=72%/52% |
| `pos905_scale7252` | 672.52% | 1.95 | 39.82% | 49.64% | 171.44% | 39.81% | pos=90.50%, dd=7.0%/12.0%, scale=72%/52% |
| `pos9025_scale7252` | 666.83% | 1.95 | 39.74% | 49.92% | 329.52% | 39.70% | pos=90.25%, dd=7.0%/12.0%, scale=72%/52% |
| `pos90_scale7555` | 713.93% | 1.96 | 40.51% | 46.67% | 171.17% | 40.41% | pos=90.00%, dd=7.0%/12.0%, scale=75%/55% |
| `pos90_dd065115_scale7555` | 709.97% | 1.96 | 40.51% | 46.67% | 171.76% | 40.29% | pos=90.00%, dd=6.5%/11.5%, scale=75%/55% |
| `pos8975_dd065115_scale7555` | 706.14% | 1.96 | 40.41% | 46.82% | 171.33% | 40.18% | pos=89.75%, dd=6.5%/11.5%, scale=75%/55% |
| `pos9025_scale7353` | nan% | nan | nan% | nan% | nan% | nan% | pos=90.25%, dd=7.0%/12.0%, scale=73%/53% |

## 硬过滤审计

- 审计文件数：`16`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所命中：`0`
- ST / 风险警示命中：`0`
- 退市命中：`0`
- 开盘涨停买入命中：`0`

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625\detail.csv`
- 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625\summary.csv`
- 准入排序：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625\summary_by_admission.csv`
- 硬过滤：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625\hard_gate_audit.json`
- 掘金日志：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625\logs`
