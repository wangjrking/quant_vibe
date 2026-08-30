# 股票风险事件 L1/L2 日更合同

## 数据范围

日常增量链新增三个 Tushare 原始接口：

- `stk_shock`：个股异常波动
- `stk_high_shock`：个股严重异常波动
- `stk_alert`：交易所重点提示证券

L1 保留原始字段，并增加 `source_api`、`source_row_sha256`、`fetched_at`。三张表分别写入独立 DuckDB 文件，不合并到日线宽表。

## 日更入口

- L1：`run_all_a_raw_update.py` 的默认表集合已包含上述三表，同时同步到 `production_assets/duckdb/l1_raw_tables/`。
- L2：`integrate_l2_stock_risk_events_daily.py --target-trade-date YYYYMMDD`。
- 标准 L1-L8 workflow 的 L2 阶段已把该 L2 脚本登记为必需 companion entrypoint。

## L2 资产

- `production_assets/duckdb/l2_stock_risk_events.duckdb::stock_risk_events`
- `production_assets/duckdb/l2_stock_risk_signal.duckdb::stock_risk_signal`

L2 信号只在源事件日之后的首个官方开市日可见。异常波动按 1/3/5/10 个交易日窗口聚合；重点提示的 active 状态按官方 `end_date`（含当日）计算。

## 固定边界

- `.BJ` 从 L1 开始排除。
- 空事件日是合法状态，不复制上一日数据。
- Tushare 调用失败或字段漂移时停止，不写入候选文件。
- 重复运行同一目标日采用目标日替换，事件和信号自然键必须唯一。
- L2 信号每天从事件明细确定性重算，且只保留截至目标交易日的切片；禁止 bootstrap 预生成未来信号日期。
- 不修改 `STOCK_DAILY_DATA`，不自动触发 L3、策略或交易。
