# formal 5D10D Top3 ST-clean 生产发布说明

## 当前结论

本版本已按用户要求发布为当前唯一 L5 生产策略。策略候选为 `top3_10d80_full_h3_nodd`，使用当前 `approved_for_l5` formal L4 5D 与 10D 预测资产，正式绩效以掘金回测日志为准。

## 策略规则

- 模型输入：当前 formal L4 5D + 10D。
- 打分：`0.8 * 10D rank + 0.2 * 5D rank`。
- 每日最多买入：Top 3。
- 持仓：3 个交易日。
- 最大持仓数：3。
- 仓位：第 1/2/3 名约 `34% / 32% / 29%`。
- 启用每日分数卖出。
- 不启用账户回撤缩仓。
- 不买北交所。
- 不买 ST / 风险警示。
- 开盘涨停不能买入。
- 不使用行业过滤。
- 不使用月份/日期过滤。
- 不使用独立 1D 模型分数。

## 掘金验证

- 年化收益：`87.83%`
- 夏普：`1.49`
- 最大回撤：`33.79%`
- 信号数：`1471`
- 买入日：`494`

## 风险与边界

本版本收益未达到 200% 年化目标，但它是当前 formal L4 5D+10D、ST-clean、非行业、非月份、非极端稀疏规则下，经掘金回测可复核的当前生产候选。后续继续优化必须另走研究链路，不能直接覆盖本生产归档。

## 证据路径

- `strategy_manifest.json`
- `trading_rules.json`
- `validation.json`
- `validation_audit.json`
- `signals/historical_top3_10d80_full_h3_nodd.csv`
- `backtests/gm_backtest_top3_10d80_full_h3_nodd.log`
- `predictions/fusion_manifest.json`
