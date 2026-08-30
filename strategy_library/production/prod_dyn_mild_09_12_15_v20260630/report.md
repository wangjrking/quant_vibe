# 动态 Top5 高收益 `dyn_mild_09_12_15` 发布说明

## 当前结论

当前正式生产策略为 `prod_dyn_mild_09_12_15_v20260630`。  
本策略当前只允许作为 `L5/L6/L7` 正式链路输入使用，不允许回退旧 SQLite、共享 DuckDB 或 research-only 资产。

## 当前正式输入

- `L4 3D formal`：`quant/data_file/production_assets/duckdb/l4_executable_3d_open_return_formal.duckdb`
- `L4 5D formal`：`quant/data_file/production_assets/duckdb/l4_executable_5d_open_return_formal.duckdb`
- `L4 10D formal`：`quant/data_file/production_assets/duckdb/l4_executable_10d_open_return_formal.duckdb`
- `L2 市场底表`：`quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb`

## 当前最新信号

- `signal_date`: `20260630`
- `buy_date`: `20260701`
- `row_count`: `5`
- `stock_count`: `5`
- `status`: `pending_buy_day_hard_gate`

## 当前边界

- 当前只完成正式信号导出与交付包准备。
- 买入日硬门控尚未完成，当前不允许解释为可自动交易执行。
- 下游若读取本策略，只能读取正式 latest signal 资产，不得读取候选表、blended 明细或历史旧库。
