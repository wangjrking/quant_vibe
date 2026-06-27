# 动态持有 Top1 三周期融合 pos74 候选归档

## 当前状态

该目录是探索区生产候选归档，状态为 `production_candidate_pending_audit_not_published`。本次未修改 `registry.json`，未替换当前生产策略，未生成生产信号。

## 核心规则

- 输入资产：formal L4 10D / 5D / 3D，均要求 `approved_for_l5`。
- 入场分数：`0.78 * rank_10d + 0.12 * rank_5d + 0.10 * rank_3d`。
- 每日最多买入：`Top1`。
- 目标仓位：`74%`。
- 持有规则：第 `2` 个交易日开始检查，满足延持条件最多持有到第 `3` 个交易日。
- 分数退出：10D 核心分数低于入场分数 `0.97`。
- 日内风控：浮亏达到 `6%` 触发止损。
- 账户回撤缩放：软触发 `6%`，硬触发 `12%`，缩放比例 `0.80 / 0.60`。
- 硬过滤：不买北交所，不买 ST / 风险警示，不买退市标记，不买开盘涨停。

## 掘金结果

- 年化收益：`373.77%`
- Sharpe：`2.04`
- 最大回撤：`36.22%`
- recent60 年化：`43.45%`
- 2026YTD 年化：`157.54%`
- 去最高 1% 代理信号后年化：`202.21%`

## 信号覆盖

- 最新候选信号日：`20260622`
- 最新候选买入日：`20260623`
- 该信号不是生产信号，不得直接交交易智能体执行。

## 残留风险

- 尚未完成审计智能体复核。
- 尚未获得用户明确发布批准。
- 尚未更新 `registry.json`，因此不是当前生产策略。
- 去最高 1% 代理信号后回撤仍会上升，尾部贡献压力需要继续披露。

## 证据路径

- `strategy_manifest.json`
- `trading_rules.json`
- `validation.json`
- `reproduction_v20260624.json`
- `predictions/prediction_table_meta.json`
- `research/validation_summary.json`
- `research/hard_gate_audit.json`
- `backtests/time_full.log`
