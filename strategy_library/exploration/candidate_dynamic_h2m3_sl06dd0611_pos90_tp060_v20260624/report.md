# dd0611 pos90 高收益贴边候选归档

## 当前结论

该版本是探索候选，不是生产策略。它在 `dd0611_s7050` 收益增强主线基础上，将目标仓位提高到 `90%`。掘金结果明显提升，但最大回撤已经接近 40% 红线，因此只能作为高收益贴边候选。

## 掘金结果

- 年化收益：`593.25%`
- Sharpe：`1.91`
- 最大回撤：`39.06%`
- recent60 年化：`51.14%`
- recent120 年化：`237.15%`
- 2026YTD 年化：`169.76%`

## 强准入状态

- 硬过滤预检通过：北交所、ST/风险警示、退市、开盘涨停买入命中均为 `0`。
- 低路径依赖预检通过：三个近期开仓锚点最差年化均为正且大于 100%。
- 贡献压力通过：去最高 1% 代理信号后年化仍为 `528.40%`。
- 残留风险：最大回撤贴近 40% 红线，不能在审计前声明生产准入通过。

## 最新信号覆盖

- 最新 signal_date：`20260622`
- 最新 buy_date：`20260623`
- 最新标的：`688059.SH` / `华锐精密`

## 证据路径

- 验证目录：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0611_pos90_tp060_validation_20260624`
- 源信号：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_dd0611_tp060_position_20260624\signals\dd0611_tp060_pos90.csv`
- 归档目录：`D:\work\quant\quant_mcp\quant\main\strategy_library\exploration\candidate_dynamic_h2m3_sl06dd0611_pos90_tp060_v20260624`
