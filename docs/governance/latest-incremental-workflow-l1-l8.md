# 最新 L1-L8 增量加工完整工作流

本文整理当前项目主线的标准增量加工链路，只描述当前应执行的正式工作流，不把 legacy 入口、历史混合链路或研究链路混入默认口径。

## 1. 适用范围

- 适用对象：L1-L8 标准生产链路
- 当前硬规则：
  - `no-BJ`：北交所股票从 L1 源头开始过滤
  - `DuckDB-only`：active 主线只认 DuckDB
  - `一张表一个文件`：active 资产按单表单文件管理
  - `qfq 显式命名`：凡前复权价格、技术字段、GTJA 因子、策略侧前复权输入，必须显式带 `_qfq`
  - `标签成熟度独立`：feature 最新日期与 label 最新成熟日期分开表述
  - `fail-closed`：任一层未完成审计或交接，不得越级推进下游

## 2. 责任分层

| 层级 | 负责智能体 | 职责 |
| --- | --- | --- |
| L1 | 数据接入智能体（data-ingestion-agent） | 源头探测、原始表接入、L1 raw DuckDB 表文件同步 |
| L2 | 数据整合智能体（data-integration-agent） | `STOCK_DAILY_DATA` 目标日增量、qfq 底表整合 |
| L3 | 因子智能体（factor-agent） | target-date feature 交付、独立 label 成熟度、no-BJ、qfq 因子契约 |
| L4 | 模型智能体（model-agent） | 基于 active L3 做目标交易日增量预测，不重训 |
| L5 | 策略智能体（strategy-agent） | 正式策略信号生成、候选筛选、latest signal 落盘 |
| L6 | 策略智能体（strategy-agent） | 策略验证、current registry / validation / backtest 同步 |
| L7 | 交易智能体（trading-agent） | 交付包生成、买入日硬门控、人工确认前检查 |
| L8 | 指挥官 / 审计 / MCP 智能体 | workflow monitor、只读审计、发布候选登记、状态网关 |

说明：

- 当前没有独立 `L9/deploy-agent` 正式线程。
- `mcp-agent` 属于 L8 终层发布与状态治理角色，不生产 L1-L7 业务资产。

## 3. 当前标准入口

| 层级 | 标准入口 | 辅助路由 / 治理入口 |
| --- | --- | --- |
| L1 | `quant/main/run_all_a_raw_update.py` | `quant/main/l1_raw_data_route.py` |
| L2 | `quant/main/integrate_l2_latest_and_recompute_qfq.py` | `quant/main/stock_daily_data_route.py` |
| L3 | `quant/main/deliver_l3_target_date_duckdb_mainline.py` | `quant/main/l3_duckdb_sync.py` |
| L4 | `quant/main/incremental_formal_l4_duckdb_mainline.py` | `quant/main/model_asset_route.py` |
| L5/L6 | `quant/main/run_production_tasks.py` | `quant/main/strategy_asset_route.py`、`l5_duckdb_sync.py`、`l6_duckdb_sync.py` |
| L7 | `quant/main/build_l7_delivery_package.py` | `quant/main/l7_buy_day_hard_gate.py`、`l7_duckdb_sync.py` |
| L8 | `quant/main/tools/workflow_monitor_manage.py` | `quant/main/tools/production_asset_gate.py` |

## 4. Active 资产终态

| 层级 | Active 资产 |
| --- | --- |
| L1 | `quant/data_file/production_assets/duckdb/l1_raw_tables/[table].duckdb` |
| L2 | `quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb::STOCK_DAILY_DATA` |
| L3 feature | `quant/data_file/production_assets/duckdb/l3_feature_current.duckdb::<active_table>` |
| L3 label | `quant/data_file/production_assets/duckdb/l3_label_current.duckdb::<active_table>` |
| L4 | 由 formal manifest 显式声明的 DuckDB prediction table |
| L5/L6 | `strategy_library` 下 production 策略归档 + DuckDB current 同步 |
| L7 | `quant/data_file/production_signals/*_latest.csv|json` + `runtime/trading_agent/delivery_packages/` |
| L8 | `runtime/orchestrator_reports/workflow_monitor/` + MCP 发布候选状态 |

## 5. 完整链路

### L1：每日增量接入

1. 以目标交易日做源头完整性复探，`daily_data` 为主驱动表。
2. 检查 9 张标准表：`daily_data`、`daily_index_data`、`stk_factor`、`moneyflow`、`limit_list_data`、`cyq_perf`、`adj_factor`、`stock_st`、`index_daily`。
3. 源头通过后，写 raw parquet 与 active L1 DuckDB 表文件。
4. 生成三面对齐证据：`source / raw parquet / active DuckDB`。
5. 等待审计通过后，才允许交接 L2。

L1 关键约束：

- 从源头开始过滤 `.BJ`
- `daily_data` 是交易日历和股票集合驱动表
- `adj_factor` 可以比 `daily_data` 多，但不得反向驱动交易日历
- `stock_st` 是事件表，不按全覆盖表解释

### L2：目标交易日增量底表

1. 读取 active L1 DuckDB 表文件，不读 legacy SQLite 作为默认输入。
2. 每次只处理 `target_trade_date`，`historical_scan=false`，禁止调用全量重建入口。
3. 写入 active L2 DuckDB：`l2_stock_daily_data.duckdb::STOCK_DAILY_DATA`。
4. 审计关注：
   - 目标交易日是否完整落入 active 表
   - 重复键是否为 0
   - no-BJ 是否真实生效
   - qfq 价格字段是否显式为 `open_qfq/high_qfq/low_qfq/close_qfq/pre_close_qfq`
5. 审计通过后交接 L3。

### L3：目标交易日 feature 交付

1. 基于 active L2 只生成目标交易日 feature；依赖窗口按最近 300 个真实交易日有界读取。
2. feature 主线必须 no-BJ、DuckDB-only、显式 qfq。
3. label 主线单独维护到最新成熟日期，不得为最新 feature 日期伪造未来标签。
4. staging 只允许位于白名单 workspace，不能进入 active route / registry / manifest。
5. 审计重点：
   - `.BJ=0`
   - 裸前复权列不存在
   - `_qfq` 技术列、GTJA 列命名完整
   - `target_date` 已写入 active feature
   - active label 未前推
6. 审计通过后交接 L4。

### L4：目标交易日增量预测

1. 只读取 active L3 DuckDB feature / label。
2. 只对目标交易日做 formal 增量预测刷新。
3. 不训练、不调参、不跑回测、不生成正式交易信号。
4. 输出写入 formal DuckDB prediction assets，并更新 formal manifest 的增量元数据。
5. 审计重点：
   - `2026-07-01` 后主线必须保持 DuckDB-only
   - manifest 必须 `approved_for_l5`
   - feature freshness 与 label maturity 分开表述
6. 审计通过后交接 L5/L6。

### L5/L6：策略信号与验证

1. 只允许读取已审批的 formal manifest。
2. 通过 `run_production_tasks.py` 执行生产策略信号导出。
3. latest signal 必须与策略归档、DuckDB current 同步一致。
4. candidate / blended score 只能留在 reports，不得冒充正式信号。
5. 审计重点：
   - `signal_date` / `buy_date` 是否正确
   - latest signal 与 full history 是否一致
   - 规则过滤导致的排除要显式留痕，不能误报为上游漏产出
6. 审计通过后交接 L7。

### L7：交易交付与买入日硬门控

1. 根据正式 latest signal 生成交付包。
2. 交付包包含：
   - `l7_platform_signals.csv`
   - `l7_delivery_summary.json`
   - `buy_day_market_snapshot_availability.json`
   - `buy_day_hard_gate_*`
3. 买入日硬门控必须优先使用实时 / 平台快照，不以收盘后 EOD 表替代。
4. 在硬门控通过前，状态只能是 `pending_buy_day_hard_gate`。
5. 硬门控通过后，最多推进到 `ready_for_human_confirmation_execution` 或 L8 的 `pending_user_approval`，仍不得解释为自动交易执行。

### L8：治理、审计、发布候选

1. 指挥官维护 workflow monitor，只登记状态，不代替各层做生产开发。
2. 审计智能体逐层只读复核，未通过不得越级。
3. MCP / 发布层只登记已审计通过的来源资产。
4. 状态语义：
   - L7 未过闸：`pending_buy_day_hard_gate`
   - L7 过闸但未人工确认：`pending_user_approval`
   - 不存在“自动可执行交易信号”直发状态

## 6. 放行规则

| 上游层 | 下游放行条件 |
| --- | --- |
| L1 -> L2 | L1 三面对齐完成，审计通过 |
| L2 -> L3 | L2 目标日底表完成，审计通过 |
| L3 -> L4 | L3 目标日 feature 与 label 成熟度合同完成，审计通过 |
| L4 -> L5/L6 | L4 formal 预测刷新完成，manifest 审计通过 |
| L5/L6 -> L7 | 正式 latest signal / status / DuckDB current 一致，审计通过 |
| L7 -> L8 | 允许进入交付与候选发布治理，但未过买入日硬门控前不得进入可执行态 |

## 7. 当前最重要的禁止事项

- 不得把 legacy SQLite、旧 shared DuckDB、`odb.db`、`stock_factor_data.parquet` 当作 active 默认输入
- 不得把 `.BJ` 留在 active 主线里再交给下游过滤
- 不得重新引入裸前复权价格或裸前复权技术字段
- 不得把 feature 最新日包装成 label 最新成熟日
- 不得在 L7 过闸前把资产登记成可自动执行
- 指挥官不得替代专业智能体执行对应生产开发动作

## 8. 最近可参考的运行证据

- L1 完整审计样例：`quant/data_file/reports/l1_incremental_raw_ingest_20260706_complete.json`
- workflow monitor 样例：`quant/data_file/runtime/orchestrator_reports/workflow_monitor/workflow_monitor_20260706_incremental-trading-signal-20260706.json`
- L3 全量交付样例：`quant/data_file/runtime/agent_workspaces/factor-agent/work/l3_full_factor_delivery_20260703/logs/l3_full_delivery_20260703.json`

## 9. 一句话版本

当前标准链路是：

`L1 每日增量 -> L2 目标日增量底表 -> L3 目标日 feature/独立 label 成熟度 -> L4 目标日增量预测 -> L5/L6 正式信号 -> L7 交付与买入日硬门控 -> L8 治理与发布候选`

日常链路采用一次主人授权和审计通过自动推进，不再逐层等待指挥官重复授权；控制线程超时只触发消息重发，不构成业务 blocked。

其中所有 active 资产都必须满足 `no-BJ + DuckDB-only + 一表一文件 + 显式 qfq`。

面向主人默认只显示四项：`状态、当前层级、业务结果、唯一下一步`。内部审计、线程、哈希和证据包继续保留，但不再作为日常沟通负担。
