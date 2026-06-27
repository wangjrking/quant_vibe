# dd0712 pos89 收益回撤改善候选归档

## 当前结论

该版本是探索候选，不是生产策略。它只在 `dd0712_tp060` 规则上把目标仓位从 `90%` 调整为 `89%`，其余 formal L4 输入、Top1 选股、动态持有、止损止盈、账户回撤缩放和硬过滤口径保持一致。

## 掘金结果

- 年化收益：`621.06%`
- Sharpe：`1.93`
- 最大回撤：`36.91%`
- recent60 年化：`50.72%`
- recent120 年化：`233.29%`
- 2026YTD 年化：`167.29%`

## 强准入状态

- 硬过滤预检通过：北交所、ST/风险警示、退市、开盘涨停买入命中均为 `0`。
- 低路径依赖预检通过：三个近期开仓锚点的最差年化均为正，且高于当前弱收益目标。
- 贡献压力预检通过：去最高 1% 代理信号后年化仍为 `552.05%`。
- 尚未通过生产发布：仍需审计复核和用户批准，不得称为当前生产策略。

## 最新信号覆盖

- 最新 signal_date：`20260622`
- 最新 buy_date：`20260623`
- 最新标的：`688059.SH` / `华锐精密`

## 证据路径

- 验证目录：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_tp060_validation_20260624`
- 源信号：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_dd0712_tp060_position_20260624\signals\dd0712_tp060_pos89.csv`
- 归档目录：`D:\work\quant\quant_mcp\quant\main\strategy_library\exploration\candidate_dynamic_h2m3_sl06dd0712_pos89_tp060_v20260624`
