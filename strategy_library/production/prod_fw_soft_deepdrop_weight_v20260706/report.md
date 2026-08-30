# 深跌低开软调仓

## 当前结论

- 策略 ID：`prod_fw_soft_deepdrop_weight_v20260714`
- 发布状态：`production`
- 当前执行状态：`pending_buy_day_hard_gate`
- `signal_date`：`20260713`
- `buy_date`：`20260714`
- 最新正式信号行数：`3`

## 生产口径

- 当前策略只读取已批准的 split DuckDB formal L4 资产。
- 当前市场底表只读取 `l2_stock_daily_data.duckdb`。
- 北交所股票不进入当前正式策略可交易范围。
- 前复权字段和技术列继续保持显式 `_qfq` 语义。

## L7 状态

- 最新正式信号已导出到生产信号目录。
- L7 交付包已生成。
- 买入日硬门控当前仍未完成，因此状态保持为 `pending_buy_day_hard_gate`。
- 在买入日快照补齐前，不允许解释为可自动执行。
