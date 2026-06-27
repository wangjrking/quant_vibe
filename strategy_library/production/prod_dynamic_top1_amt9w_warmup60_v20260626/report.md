# 单票动态Top1 量额95k dd08_115

## 当前结论

这是当前已通过策略准入的生产替换版。它沿用 formal L4 3D/5D/10D 模型输入，不使用 1D，不使用行业/月度/日期限制，只在单票动态 Top1 链路上把候选池收紧到 `amount >= 95000`，并把账户回撤软阈值提升到 `8%`、硬阈值保持 `11.5%`。

## 核心规则

- 模型打分：`0.78 * rank_10d + 0.12 * rank_5d + 0.10 * rank_3d`
- 候选过滤：不买北交所、不买 ST/风险警示、不买退市、不买开盘涨停
- 流动性过滤：`amount >= 95000`，`total_mv >= 200000`
- 持仓规则：单票，目标仓位 `89.75%`
- 持有规则：至少 2 天，最多 3 天，`score_exit_entry_ratio = 0.97`，`score_continue_entry_ratio = 0.975`
- 风控规则：止损 `6%`，止盈 `7%`，账户回撤 soft/hard `8% / 11.5%`，缩放 `75% / 55%`

## 掘金验证

- 全周期年化：`950.65%`
- 全周期累计收益：`1950.79%`
- Sharpe：`2.10`
- 最大回撤：`39.26%`
- recent120 年化：`286.69%`
- recent60 年化：`46.82%`
- 2026YTD 年化：`206.57%`
- 平均持仓率：`41.09%`

## 低路径依赖复核

- 锚点 `20250701`：近邻空仓启动最差年化 `384.27%`，中位年化 `386.68%`
- 锚点 `20251009`：近邻空仓启动最差年化 `206.16%`，中位年化 `212.90%`
- 锚点 `20260105`：近邻空仓启动最差年化 `191.44%`，中位年化 `237.12%`

## 最新生产信号

- signal_date：`20260624`
- buy_date：`20260625`
- buy_day_hard_gate_complete：`True`

## 证据路径

- 验证目录：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_amt9p5w_s7555_dd08_115_current_validation_20260625`
- 全量信号：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_prod_filter_amount_interpolate_20260625\signals\prod_filter_amt9p5w_mv20w.csv`
- 最新信号：`D:\work\quant\quant_mcp\quant\data_file\production_signals\prod_dynamic_top1_amt9p5w_dd08_115_v20260625_latest.csv`
