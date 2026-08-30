# `prod_dyn_mild_09_12_15_v20260630` 当前策略说明

## 策略身份

- `strategy_id`: `prod_dyn_mild_09_12_15_v20260630`
- `status`: `production`
- `universe`: `no-BJ`
- `execution_state`: `pending_buy_day_hard_gate`

## 当前主链约束

- 只允许读取 split DuckDB formal 预测资产。
- 只允许读取 split DuckDB 的 `L2 STOCK_DAILY_DATA`。
- 前复权价格、技术字段和 GTJA 因子必须显式带 `_qfq`。
- 原始市场价格输出必须保持 `*_raw` 语义，不得混淆为前复权值。

## 当前最新信号状态

- `signal_date`: `20260630`
- `buy_date`: `20260701`
- `latest_signal_status`: `pending_buy_day_hard_gate`
- `buy_day_hard_gate_complete`: `false`
- `buy_day_realtime_checks_required`: `true`

## 风险与边界

- 当前状态只表示正式 latest signal 已生成。
- 当前不表示买入日硬门控已通过。
- 当前不表示允许自动交易执行。
- 若后续出现下游排除，必须显式记录为下游规则排除，不得误写成上游 DuckDB 主链缺失。
