# 深跌低开软调仓

## 当前状态

- 策略 ID：`prod_fw_soft_deepdrop_weight_v20260714`
- 当前正式状态：`production`
- 当前门控状态：`pending_buy_day_hard_gate`
- `signal_date`：`20260713`
- `buy_date`：`20260714`
- 最新正式信号股票数：`3`

## 输入契约

- L4 输入来自已批准的 formal manifest，对应 split DuckDB 预测表。
- L2 市场底表来自 `l2_stock_daily_data.duckdb`。
- 不允许回退到 legacy SQLite、共享 DuckDB 或历史研究资产。

## 风险说明

- 当前只是完成增量信号导出与 L7 交付准备。
- 买入日实时行情尚未完成硬门控。
- 在硬门控通过前，必须维持 `pending_buy_day_hard_gate`，不得视为可自动交易。
