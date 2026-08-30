# 生产策略复跑报告 20260704

## 固定规则

后续复跑 `prod_repro500_p435_s96_p10d70_v20260702` 必须遵守以下规则：

1. 只允许从本生产归档读取：
   - `signals/full_history_repro500_p435_s96_p10d70.csv`
   - `inputs/score.duckdb`
   - `code_snapshot/main.py`
   - `trading_rules.json`
   - `validation.json`
2. 不允许从 `quant/data_file/reports/` 下的研究网格、临时信号、临时 score 库默认取数。
3. 必须显式传入 `max_positions=3`，不得依赖代码默认值。
4. 回测执行价格口径为不复权，`backtest_adjust=none`。
5. 当前归档复跑口径关闭自适应滑点，使用 `backtest_slippage_ratio=0.0`。
6. 研究目录结果只能作为研究证据，不能覆盖生产复跑结论。

## 本次复跑命令要点

- 策略代码：`code_snapshot/main.py`
- 信号文件：`signals/full_history_repro500_p435_s96_p10d70.csv`
- 分数库：`inputs/score.duckdb::score`
- 行情库：`quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb`
- `max_positions=3`
- `backtest_adjust=none`
- 初始资金：`700000`
- 滑点：`0.0`

## 本次掘金复跑结果

- 回测窗口：`2022-08-12 09:00:00` 到 `2026-06-08 15:30:00`
- 累计收益：`1539.97%`
- 年化收益：`402.35%`
- Sharpe：`4.458`
- 最大回撤：`3.25%`
- 开仓 / 平仓：`296 / 296`
- 胜率：`80.07%`

## 与旧发布结果差异

旧发布日志记录：

- 年化收益：`500.50%`
- Sharpe：`3.906`
- 最大回撤：`3.87%`
- 开仓 / 平仓：`279 / 279`

本次复跑没有完全复现旧年化，但已经确认此前低结果的主要错误是漏传 `max_positions=3`，导致只按单持仓执行。

当前应采用本次生产归档内复跑结果作为最新可复跑结果。旧 `500.50%` 记录保留为历史发布日志，不再作为未复核的当前展示口径。

## 证据

- 复跑合同：`production_replay_contract_20260704.json`
- 复跑日志：`backtests/repro500_production_rerun_20260704_maxpos3.log`
