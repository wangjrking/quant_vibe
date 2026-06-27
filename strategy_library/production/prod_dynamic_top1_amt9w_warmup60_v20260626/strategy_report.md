# 单票动态Top1 量额90k warmup60

## 当前结论

本次发布把最新 formal L4 生产模型资产下重新优化后的单票 Top1 策略提升为当前 L5 生产策略。
该版本的关键变化不是再追求更高的静态收益，而是把 `warm-up 60` 写成正式接入口径，使其在低路径依赖准入上可交付。

## 核心规则

- 模型输入：formal L4 3D / 5D / 10D
- 打分权重：`10D 0.84 + 5D 0.09 + 3D 0.07`
- 过滤：`amount >= 90000`，`total_mv >= 200000`
- 硬门：不买北交所、不买 ST/风险警示、不买退市、不买开盘涨停
- 持仓：单票，目标仓位 `90%`
- 持有：至少 `2` 天，最多 `3` 天
- 分数续持/卖出：`0.97 / 0.97`
- 风控：止损 `6%`，止盈 `7%`，账户回撤 `8.5% / 12%` 缩放
- 接入要求：新账户默认先做 `60` 个交易日 warm-up 状态重建

## 掘金验证

- 全周期年化：`1010.07%`
- 全周期累计收益：`1992.91%`
- 夏普：`1.59`
- 最大回撤：`39.75%`
- recent120 年化：`142.03%`
- recent60 年化：`50.17%`
- 2026YTD 年化：`87.09%`
- 平均持仓率：`37.34%`

## 低路径依赖

以下为 `warm-up 60` 接入口径下，三个锚点及其相邻启动日的最差结果：
- 锚点 `20250701`：近邻最差年化 `251.84%`，中位年化 `257.18%`
- 锚点 `20251009`：近邻最差年化 `196.02%`，中位年化 `209.19%`
- 锚点 `20260105`：近邻最差年化 `118.38%`，中位年化 `169.74%`

## 硬门审计

- 信号覆盖：`499/499` 交易日，无断档
- 北交所命中：`0`
- ST/风险警示命中：`0`
- 退市命中：`0`，退市命名命中：`0`
- 开盘涨停买入命中：`0`
- 最新信号：`signal_date=20260625`，`buy_date=pending`，`buy_day_hard_gate_complete=False`

## 证据路径

- round3 汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_latest_formal_prod_dynamic_top1_opt_round3_20260626\summary.csv`
- 低路径依赖汇总：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_latest_formal_prod_dynamic_top1_lowpath_20260626\summary.csv`
- warmup60 邻域复核：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_latest_formal_prod_dynamic_top1_w84_warmup60_nearby_20260626\nearby_warmup60_summary.csv`
- 当前发布验证目录：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_publish_dynamic_top1_amt9w_warmup60_20260626`
