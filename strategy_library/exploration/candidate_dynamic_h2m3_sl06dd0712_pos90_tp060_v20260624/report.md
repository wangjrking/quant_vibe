# dd0712 pos90 收益回撤改善候选归档

## 当前结论

该版本是探索候选，不是生产策略。它在 `pos90` 高收益主线基础上，将账户回撤缩放调整为 `soft=7% / hard=12% / scale=0.70/0.50`。掘金结果显示年化提升且最大回撤下降，但仍需审计复核。

## 掘金结果

- 年化收益：`615.04%`
- Sharpe：`1.90`
- 最大回撤：`37.25%`
- recent60 年化：`51.14%`
- recent120 年化：`237.15%`
- 2026YTD 年化：`169.19%`

## 强准入状态

- 硬过滤预检通过：北交所、ST/风险警示、退市、开盘涨停买入命中均为 `0`。
- 低路径依赖预检通过：三个近期开仓锚点最差年化均为正且大于 100%。
- 贡献压力通过：去最高 1% 代理信号后年化仍为 `546.17%`。
- 未通过生产发布：尚未审计复核，尚未写入生产 registry。

## 最新信号覆盖

- 最新 signal_date：`20260622`
- 最新 buy_date：`20260623`
- 最新标的：`688059.SH` / `华锐精密`

## 证据路径

- 验证目录：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos90_tp060_validation_20260624`
- 源信号：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos90_tp060_dd_neighborhood_20260624\signals\tp060_pos90_dd0712_s7050.csv`
- 归档目录：`D:\work\quant\quant_mcp\quant\main\strategy_library\exploration\candidate_dynamic_h2m3_sl06dd0712_pos90_tp060_v20260624`
