# pos8975_scale7555 研究候选归档

## 当前结论

该版本是探索候选，不是生产策略。它在 c975_tp070 基础上只微调目标仓位和账户回撤缩放：目标仓位 `89.75%`，账户回撤 soft/hard 缩放 `75%/55%`。

生产 registry 未修改，正式发布仍需要用户批准和审计复核。

## 掘金验证结果

- 年化收益：`745.91%`
- 累计收益：`1530.64%`
- Sharpe：`1.97`
- 最大回撤：`39.25%`
- recent60 年化：`46.82%`
- recent120 年化：`238.75%`
- 2026YTD 年化：`170.76%`
- 平均仓位：`40.39%`

## 硬过滤审计

- 审计文件数：`6`
- 失败文件数：`0`
- 买入日行情缺失：`0`
- 北交所命中：`0`
- ST / 风险警示命中：`0`
- 退市命中：`0`
- 开盘涨停买入命中：`0`

## 最新研究信号

- 最新 signal_date：`20260622`
- 最新 buy_date：`20260623`
- 最新标的：`688059.SH` / `华锐精密`

## 证据路径

- 验证目录：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos8975_scale7555_validation_20260625`
- 边界搜索目录：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625`
- 源信号：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625\signals\pos8975_scale7555.csv`
- 归档目录：`D:\work\quant\quant_mcp\quant\main\strategy_library\exploration\candidate_dynamic_h2m3_sl06dd0712_pos8975_c975_tp070_scale7555_v20260625`
