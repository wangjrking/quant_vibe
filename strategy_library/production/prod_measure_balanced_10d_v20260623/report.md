# 稳健均衡10D生产发布说明

## 2026-06-23 状态修正

该版本不得发布为可交易生产策略。

原因：策略侧按项目硬门禁重新复核 `signals/historical_repro_current_best.csv`，共 1910 条历史信号中有 627 条在 `signal_date` 与 `buy_date` 均命中 `STOCK_DAILY_DATA.ST_TYPE='ST'` / `STOCK_DAILY_DATA.ST_TYPE_name='风险警示板'`。因此，虽然该历史信号在掘金复跑中记录了年化约 412.96%、夏普约 3.11、最大回撤约 20.11%，但该收益结果不能作为当前生产可交易版本发布。

当前处理：`strategy_manifest.json` 已改为 `status=blocked`，`strategy_library/registry.json` 已取消当前生产策略指向，`production_tasks.json` 保持 `enabled=false`。

## 当前结论

`prod_measure_balanced_10d_v20260623` 原拟提升为 L5 生产策略，但因 ST/风险警示硬门禁未通过，当前仅保留为阻塞归档和审计复现证据。

本次发布依据为旧 `repro_current_best.csv` 同口径掘金复跑结果：年化约 412.96%，夏普约 3.11，最大回撤约 20.11%，平均持仓率约 80.11%。

## 策略规则

该策略来自 L5 分层融合、弱日池、风险重加权链路。组合最多约 7 只股票，启用每日评分卖出、账户回撤缩仓、北交所/ST/退市/涨停过滤，以及开盘涨停不买入约束。

## 生产边界

当前生产发布只确认历史验证版本，不确认 `balanced_10d_formal_l4_20240604_20260622.csv` 为同口径延展信号。该延展信号与旧 `historical_repro_current_best.csv` 在历史窗口交集为 0，且掘金复跑年化约 6.23%、夏普约 0.216，因此已标记为不可用。

## 证据路径

- `backtests/summary_by_objective.csv`
- `backtests/repro_current_best.log`
- `signals/historical_repro_current_best.csv`
- `signals/latest_signal_status.json`

## 当前阻塞

生产策略已发布，但最新交易信号需要恢复或重建 `repro_current_best` 同口径 L5 exporter 后再生成。自动化任务已保持关闭，避免误用非同口径信号。
