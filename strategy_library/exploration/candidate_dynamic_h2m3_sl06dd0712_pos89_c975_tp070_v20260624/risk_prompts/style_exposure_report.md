# pos89 风格暴露和风格漂移审计

## 当前结论

- 审计对象：`continue_c975`。
- 是否触发风格风险提示：`True`。
- 最大单一风格集中度：`94.96%`。
- 出现主导风格漂移的维度：mv_bucket, amount_bucket, turnover_bucket, atr_bucket, price_bucket, industry。

该审计只使用买入日当时的 `STOCK_DAILY_DATA` 字段，不使用最新状态回填历史样本。风格暴露和风格漂移仅作为风险提示，不单独阻断生产准入。

## 全周期主导风格

| 维度 | 主导分组 | 占比 |
| --- | --- | ---: |
| `mv_bucket` | `mv_q5_high` | 33.47% |
| `amount_bucket` | `amount_q5_high` | 46.17% |
| `turnover_bucket` | `turnover_q5_high` | 40.73% |
| `atr_bucket` | `atr_missing` | 83.06% |
| `price_bucket` | `price_5_10` | 24.40% |
| `age_bucket` | `old_ge750d` | 94.96% |
| `industry` | `软件服务` | 9.27% |

## 窗口漂移

| 窗口 | 维度 | 主导分组 | 占比 | 分组数量 |
| --- | --- | --- | ---: | ---: |
| `full` | `mv_bucket` | `mv_q5_high` | 33.47% | 5 |
| `full` | `amount_bucket` | `amount_q5_high` | 46.17% | 5 |
| `full` | `turnover_bucket` | `turnover_q5_high` | 40.73% | 5 |
| `full` | `atr_bucket` | `atr_missing` | 83.06% | 6 |
| `full` | `price_bucket` | `price_5_10` | 24.40% | 5 |
| `full` | `age_bucket` | `old_ge750d` | 94.96% | 4 |
| `full` | `industry` | `软件服务` | 9.27% | 75 |
| `recent120` | `mv_bucket` | `mv_q4` | 25.00% | 5 |
| `recent120` | `amount_bucket` | `amount_q4` | 30.83% | 5 |
| `recent120` | `turnover_bucket` | `turnover_q4` | 25.83% | 5 |
| `recent120` | `atr_bucket` | `atr_missing` | 78.33% | 6 |
| `recent120` | `price_bucket` | `price_ge50` | 24.17% | 5 |
| `recent120` | `age_bucket` | `old_ge750d` | 95.00% | 4 |
| `recent120` | `industry` | `电气设备` | 11.67% | 44 |
| `recent60` | `mv_bucket` | `mv_q5_high` | 30.00% | 5 |
| `recent60` | `amount_bucket` | `amount_q4` | 31.67% | 4 |
| `recent60` | `turnover_bucket` | `turnover_q2` | 26.67% | 5 |
| `recent60` | `atr_bucket` | `atr_missing` | 66.67% | 6 |
| `recent60` | `price_bucket` | `price_ge50` | 30.00% | 5 |
| `recent60` | `age_bucket` | `old_ge750d` | 91.67% | 3 |
| `recent60` | `industry` | `汽车配件` | 15.00% | 26 |
| `recent20` | `mv_bucket` | `mv_q5_high` | 50.00% | 5 |
| `recent20` | `amount_bucket` | `amount_q5_high` | 45.00% | 4 |
| `recent20` | `turnover_bucket` | `turnover_q5_high` | 50.00% | 5 |
| `recent20` | `atr_bucket` | `atr_q5_high` | 40.00% | 6 |
| `recent20` | `price_bucket` | `price_ge50` | 40.00% | 5 |
| `recent20` | `age_bucket` | `old_ge750d` | 90.00% | 3 |
| `recent20` | `industry` | `汽车配件` | 15.00% | 15 |

## 证据路径

- 明细：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624\style_exposure_audit_20260624\style_enriched_signals.csv`
- 全周期分布：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624\style_exposure_audit_20260624\style_distribution.csv`
- 候选一致性：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624\style_exposure_audit_20260624\style_case_summary.csv`
- 窗口漂移：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624\style_exposure_audit_20260624\style_drift_summary.csv`
- 结论 JSON：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624\style_exposure_audit_20260624\style_warning_summary.json`
