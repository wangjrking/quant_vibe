# c975_tp070 收益候选归档

## 当前结论

该版本是探索候选，不是生产策略。它沿用 `pos89/sl06/dd0712/c975` 主规则，只把止盈从 `6%` 调整为 `7%`。生产 registry 未修改，正式发布仍需用户批准和审计复核。

## 掘金验证结果

- 年化收益：`700.07%`
- Sharpe：`1.99`
- 最大回撤：`36.91%`
- recent60 年化：`50.72%`
- recent120 年化：`233.29%`
- 2026YTD 年化：`167.29%`

## 强准入状态

- 硬过滤预检通过：北交所、ST/风险警示、退市、开盘涨停买入命中均为 `0`。
- 低路径依赖预检通过：三个近期开仓锚点最低年化均为正，且没有复现前期冷启动崩塌。
- 贡献压力预检通过：去最高 1% 代理信号后年化仍为 `611.20%`。
- 风格暴露和风格漂移仅作为提示披露，不单独阻断生产准入。
- 风格提示：`True`；最大单一风格集中度 `94.96%`。
- 尚未发布生产：仍需审计复核和用户明确批准。

## 最新研究信号覆盖

- 最新 signal_date：`20260622`
- 最新 buy_date：`20260623`
- 最新标的：`688059.SH` / `华锐精密`

## 证据路径

- 验证目录：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_tp070_validation_20260624`
- 窄邻域目录：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_fine_neighborhood_20260624`
- 风格提示目录：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_rule_neighborhood_20260624\style_exposure_audit_20260624`
- 源信号：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_fine_neighborhood_20260624\signals\c975_tp070.csv`
- 归档目录：`D:\work\quant\quant_mcp\quant\main\strategy_library\exploration\candidate_dynamic_h2m3_sl06dd0712_pos89_c975_tp070_v20260624`
