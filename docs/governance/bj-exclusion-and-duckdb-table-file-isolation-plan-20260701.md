> 执行状态更新（2026-07-02）：active production 已切到 no-BJ + split DuckDB-only。本文后续出现的 `quant_production.duckdb`、`STOCK_DAILY_DATA.db`、`MODEL_PREDICTIONS.db`、`odb.db`，除非明确写为 current route 复核结论，否则均按 2026-07-01 方案制定时的问题描述、过渡态降级规则或历史残留处置边界理解。
# 会话来自：架构师智能体（architect-agent）

## 同步类型

架构评审

## 任务编号

`project-rule-change-20260701-architecture`

## 评审对象

项目级规则变更：

1. 本项目不再覆盖北交所股票，L1 起始口径即排除 `.BJ`
2. DuckDB 生产资产从“单文件多表”改为“一张表一个 DuckDB 文件”

## 日期口径

- 规则生效时间：`2026-07-01`
- 目标回看工作流：`20260630`

## 决策依据

本次方案以 `D:\work\quant\quant_mcp\quant\data_file\runtime\agent_memory\decisions.json` 中 `decision-20260701-018` 为唯一项目级指令来源。该决策已明确两条硬规则：

1. 本项目股票池不再包含北交所股票，L1 源头接入必须从源头过滤 `.BJ`
2. DuckDB 生产资产从单文件多表调整为一张表一个 DuckDB 文件，降低同文件多表锁冲突风险

## 2026-07-01 制定方案时的观察

当前主线架构 **不满足** 新规则，必须分阶段整改后再送审：

1. `.BJ` 记录已进入当前 L1-L4 主线资产，不是单点遗留问题，必须按层清理并重建下游
2. 方案制定当时 DuckDB 主线仍以 `quant_production.duckdb` 单文件多表为核心，不符合文件隔离规则
3. 本轮应先固化治理方案，再由各责任智能体分层整改、逐层送审；不得直接删数据或切换 registry

## 现状证据

### 1. `.BJ` 已进入当前主线资产

- `D:\work\quant\quant_mcp\quant\data_file\stock_pool_all_a.csv`
  - 当前仍包含 `.BJ` 股票，抽样统计为 `322` 行
- `D:\work\quant\quant_mcp\quant\data_file\raw_table_dbs\daily_data.DB::daily_data`
  - 当前仍包含 `.BJ` 记录，读数为 `325573` 行
- `D:\work\quant\quant_mcp\quant\data_file\raw_table_dbs\stock_basic_data.DB::stock_basic_data`
  - 当前仍包含 `.BJ` 记录，读数为 `322` 行
- `D:\work\quant\quant_mcp\quant\data_file\production_assets\duckdb\quant_production.duckdb`
  - `daily_data` 仍含 `.BJ`
  - `STOCK_DAILY_DATA` 仍含 `.BJ`
  - `prod_l3_production_factor_parts_20260625` 仍含 `.BJ`
  - `prod_l3_prediction_label_parts_current` 仍含 `.BJ`
  - L4 formal 3D/5D/10D 表仍含 `.BJ`

### 2. 20260630 工作流已受 `.BJ` 影响

- `D:\work\quant\quant_mcp\quant\data_file\runtime\orchestrator_reports\workflow_monitor\workflow_monitor_20260630_incremental-trading-signal-20260630.json`
  - 已出现 `920222.BJ` 相关 probe、L2 任务记录和 L4 风险记录
  - 说明 `.BJ` 已进入增量工作流，不是只存在于历史备份

### 3. 方案制定当时 DuckDB 主线仍是单文件多表

- `D:\work\quant\quant_mcp\quant\main\duckdb_asset_route.py`
  - 制定方案时 `DUCKDB_PRODUCTION_FILE` 固定为 `quant_production.duckdb`
  - 当前各层路径解析仍以同一个生产 DuckDB 文件为中心
- `D:\work\quant\quant_mcp\quant\data_file\asset_registry\production_assets.json`
  - 制定方案时 L1/L2/L3/L4/L5/L7 生产资产都指向 `quant\data_file\production_assets\duckdb\quant_production.duckdb` 或其 `::table` 形式
- `D:\work\quant\quant_mcp\WORKFLOW.md`
- `D:\work\quant\quant_mcp\quant\main\AGENTS.md`
  - 制定方案时文档口径仍把单文件 DuckDB 视为主线生产存储

## 新规则边界

### A. 股票池边界

自 `2026-07-01` 起，生产主线仅覆盖沪深 A 股，不再覆盖北交所股票。

硬规则如下：

1. L1 源头拉取、源头探针、股票池导出、driver universe 都不得再把 `.BJ` 当成主线样本
2. `.BJ` 不能再通过下游 SQL 条件临时过滤来维持生产正确性，必须上移到 L1 规则层
3. 历史 `.BJ` 数据可在 rollback / archive / evidence 中保留，不得继续作为默认主线输入
4. 任何新生产资产若仍含 `.BJ`，审计应直接阻断

### B. DuckDB 文件隔离边界

自本次方案执行后，生产 DuckDB 不再允许“单文件多表主线”。

硬规则如下：

1. 一张生产表对应一个 DuckDB 文件
2. 文件名与正式表名一致
3. route / registry / manifest 必须显式指向具体文件和表
4. 旧 `quant_production.duckdb` 只能作为 rollback / archive 观察资产，不得继续扩张为新主线
5. 实验 DuckDB 文件不得复用生产目录

### C. 前复权字段命名边界

自本次治理起，凡是前复权价格字段，以及基于前复权价格计算的技术字段、衍生字段、因子和策略输入字段，都必须显式标记 `qfq` 口径。

硬规则如下：

1. 若字段实际值为前复权，则字段名必须显式体现 `qfq`
2. 未复权、前复权、后复权及其他 adjustment semantics 不得混用裸字段名
3. 策略、模型、交易、审计和文档层都不得依赖“约定俗成”去猜字段口径
4. 值为前复权但字段名未显式标记的情况，一律视为治理缺陷
5. 该缺陷在 L2/L3/L4/L5 审计中属于可阻断缺陷，不是文档备注项

## 目标存储布局

建议统一到以下生产路径根目录：

`D:\work\quant\quant_mcp\quant\data_file\production_assets\duckdb\production\`

按层分目录、按表分文件：

### L1

- `.../l1/daily_data.duckdb::daily_data`
- `.../l1/daily_basic.duckdb::daily_basic`
- `.../l1/adj_factor.duckdb::adj_factor`
- `.../l1/moneyflow.duckdb::moneyflow`
- `.../l1/stk_factor.duckdb::stk_factor`
- `.../l1/stock_basic_data.duckdb::stock_basic_data`
- 其余正式 raw 表均同规则处理

### L2

- `.../l2/STOCK_DAILY_DATA.duckdb::STOCK_DAILY_DATA`

### L3

- `.../l3/prod_l3_production_factor_parts_20260625.duckdb::prod_l3_production_factor_parts_20260625`
- `.../l3/prod_l3_prediction_label_parts_current.duckdb::prod_l3_prediction_label_parts_current`

### L4

- `.../l4/stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal.duckdb::stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal`
- `.../l4/stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal.duckdb::stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal`
- `.../l4/stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal.duckdb::stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal`

### L5

- `.../l5/prod_l5_strategy_registry_current.duckdb::prod_l5_strategy_registry_current`
- `.../l5/prod_l5_strategy_manifest_current.duckdb::prod_l5_strategy_manifest_current`

### L6

- `.../l6/prod_l6_strategy_validation_current.duckdb::prod_l6_strategy_validation_current`
- `.../l6/prod_l6_backtest_files_current.duckdb::prod_l6_backtest_files_current`
- `.../l6/prod_l6_backtest_json_current.duckdb::prod_l6_backtest_json_current`
- `.../l6/prod_l6_backtest_csv_rows_current.duckdb::prod_l6_backtest_csv_rows_current`

### L7

- `.../l7/prod_l7_signal_rows_current.duckdb::prod_l7_signal_rows_current`
- `.../l7/prod_l7_signal_status_current.duckdb::prod_l7_signal_status_current`
- `.../l7/prod_l7_signal_files_current.duckdb::prod_l7_signal_files_current`

### L8

L8 不生产这些表，但所有发布契约、registry、manifest、MCP 对外说明必须按“单表单文件”口径引用上游正式资产。

## 立即需要整改的层

### L1：立即整改，优先级最高

责任：`data-ingestion-agent`

必须改：

1. `quant/main/stock_pool_module.py`
   - 取消生产主线对 `.BJ` 的包含能力
   - `build_all_a_stock_pool(... include_bj=True)` 不应继续作为生产默认口径
2. `quant/main/run_all_a_raw_update.py`
   - source preflight、driver universe、stock pool 解析必须与“排除 `.BJ`”一致
3. `quant/main/raw_table_db_module.py`
4. `quant/main/l1_duckdb_sync.py`
5. `backfill_daily_tables_from_tushare.py`
6. `backfill_extension_tables_from_tushare.py`
7. `run_0609_dual_signals.py`
   - 以上写链如果仍会把 `.BJ` 写入 L1 正式资产，必须阻断
8. L1 正式 DuckDB 由“单文件多表”改为“每张表一个 DuckDB 文件”

产出要求：

1. 新 L1 正式 raw 资产不含 `.BJ`
2. L1 新 DuckDB 路由不再指向 `quant_production.duckdb`
3. 保留旧 split DB / 旧单库 DuckDB 作为 rollback，不得删除

### L2：必须全量重建

责任：`data-integration-agent`

原因：

1. 当前 `STOCK_DAILY_DATA` 已含 `.BJ`
2. L2 依赖 L1 组合口径，不能只做按日删补
3. 单库 DuckDB 方案也必须改为 `STOCK_DAILY_DATA.duckdb`
4. 当前若 `open/high/low/close/pre_close` 实际承载前复权值，但字段名未显式标注 `qfq`，则属于当前治理缺陷，不能继续带入终态架构

必须改：

1. `quant/main/stock_daily_data_route.py`
2. `quant/main/l2_duckdb_sync.py`
3. `quant/main/run_incremental_cdb_update.py`
4. `quant/main/data_process_module.py`
5. `quant/data_file/backfill_parts/integrate_stock_daily_target_date_from_split_raw_*.py`

口径收口要求：

1. 终态优先方案：将前复权字段显式改名为 `open_qfq/high_qfq/low_qfq/close_qfq/pre_close_qfq`
2. 若短期不能改名，则只能作为过渡态治理例外，必须同时补：
   - 强约束 schema
   - 强约束 manifest
   - 强约束 route metadata
   - 所有裸字段歧义引用清理清单
3. 在终态文档、主线 manifest、主线 route 说明中，不得继续用裸字段隐含表达前复权语义

产出要求：

1. 全量重建无 `.BJ` 的 L2 正式资产
2. 新 route 指向 `.../l2/STOCK_DAILY_DATA.duckdb`
3. old SQLite / old single-file DuckDB 只保留 rollback 角色

### L3：必须全量重建

责任：`factor-agent`

原因：

1. 当前 features / labels 都已含 `.BJ`
2. L3 按截面与标签口径构造，不能用局部 delete 替代正式重建
3. 凡基于前复权价格生成的技术字段、滚动特征和因子，终态必须显式带 `qfq` 语义，不能继承 L2 裸字段歧义

必须改：

1. `quant/main/build_production_factor_parts.py`
2. `quant/main/build_prediction_label_parts.py`
3. `quant/main/rebuild_factor_data_batched.py`
4. `quant/main/rebuild_raw_factor_by_stock.py`
5. `quant/main/run_incremental_factor_update_chunked.py`
6. `quant/main/gtja_alpha_workflow.py`
7. L3 route / sync / bypass scan 工具

产出要求：

1. features / labels 全量重建无 `.BJ`
2. 每张正式表单独 DuckDB 文件
3. Parquet 仅保留 rollback / legacy / research 角色
4. L3 schema、字段字典、构建脚本和审计报告必须显式写清 adjustment semantics

### L4：必须重做输入路由与 formal manifest

责任：`model-agent`

原因：

1. 当前 L4 formal 表已含 `.BJ`
2. 当前 formal manifest 仍建立在旧 L3 产物与单库 DuckDB 假设之上
3. 当前模型输入若依赖前复权值或基于前复权的因子，但 manifest 未显式声明 adjustment semantics，则属于治理缺陷

必须改：

1. `quant/main/model_asset_route.py`
2. `quant/main/prediction_manifest.py`
3. `quant/main/predict_saved_models_standard_chain.py`
4. `quant/main/refit_fold09_predict_20260616_production_chain.py`
5. `quant/main/rolling_train_module.py`
6. 三份 active formal manifests

输入契约要求：

1. L4 manifest 必须显式写明所用价格口径和 adjustment semantics
2. 若读取 L2 价格字段，必须明确该字段是 `qfq`、未复权或其他口径
3. 若读取 L3 因子，必须明确其是否由 `qfq` 价格派生
4. 不允许模型消费者靠历史习惯猜测字段口径

边界：

1. 本轮架构方案不授权直接训练
2. 待 L3 新正式资产审计通过后，L4 再按新输入口径做增量或全量发布

### L5 / L6 / L7：立即改读路由与历史说明，不先改生产逻辑

责任：`strategy-agent`、`trading-agent`

必须改：

1. `quant/main/strategy_asset_route.py`
2. `quant/main/run_production_tasks.py`
3. `quant/main/export_dynamic_top1_formal_signals.py`
4. `quant/main/export_gm_signals.py`
5. 相关 strategy manifest / signal manifest / delivery docs

要求：

1. 只允许读取新的 L4/L5/L6/L7 单表单文件 DuckDB 正式资产
2. 历史目录型资产与旧单库 DuckDB 必须降级为 rollback / evidence
3. 任何仍假定 `quant_production.duckdb` 为默认主线的逻辑都要收口
4. 策略输入契约、信号生成说明、交易交付说明中，必须显式写清使用的价格 adjustment semantics；若信号依赖 `qfq` 口径，则必须在字段名、manifest 或 contract 中明确标注

### L8：更新发布契约与对外说明

责任：`mcp-agent`

必须改：

1. MCP 发布层文档
2. L8 registry / manifest / asset publication docs
3. 对外资产描述中的旧股票池和旧单库 DuckDB 口径

要求：

1. L8 只引用新的正式资产，不再引用北交所口径
2. L8 对外契约必须明确单表单文件路径和审计记录来源

## 必须先改的 active route / registry / manifest / path

以下对象属于“先改入口、后谈重建”的硬顺序，未改完前不得宣称新规则已落地。

### 第一优先级：active route

这些 route 决定默认主线实际读写入口，必须最先收口：

1. `quant/main/duckdb_asset_route.py`
   - 从“按层映射到单库”改为“按表映射到单文件”
2. `quant/main/l1_raw_data_route.py`
   - active L1 不得再默认回指含 `.BJ` 的旧资产
3. `quant/main/stock_daily_data_route.py`
   - active L2 必须改为解析 `STOCK_DAILY_DATA.duckdb`
4. `quant/main/model_asset_route.py`
   - active L3/L4 必须改为解析单表单文件 DuckDB
5. `quant/main/strategy_asset_route.py`
   - active L5/L6/L7 必须改为解析新的 DuckDB bundle 或 companion files
6. `quant/main/prediction_manifest.py`
   - active L4 formal manifest 必须写明新 `db_path` 和 `table`

### 第二优先级：active registry

以下 registry 条目必须在对应层审计通过后才切换，且切换时必须与 route 一起生效：

1. `quant/data_file/asset_registry/production_assets.json`
   - L1 active 条目从单库 DuckDB 或 split DB 迁移到各表单文件 DuckDB
   - L2 active 条目迁移到 `STOCK_DAILY_DATA.duckdb`
   - L3 active 条目迁移到 feature / label 各自单文件
   - L4 active 条目迁移到 3D/5D/10D formal 单表单文件或等价 bundle 契约
   - L5/L6/L7 active 条目迁移到新 bundle / companion file 结构
2. 任何旧 active 条目降级时，必须同时写：
   - `status=legacy_rollback` 或 `evidence_only`
   - `allowed_for_main_workflow=false`
   - `superseded_by=<new asset id>`

### 第三优先级：active manifest

以下 manifest 属于主线契约，不得晚于 route 切换：

1. `quant/main/config/prediction_manifests/executable_3d_open_return_l4_formal_20260617.json`
2. `quant/main/config/prediction_manifests/executable_5d_open_return_l4_formal_20260620.json`
3. `quant/main/config/prediction_manifests/executable_10d_open_return_l4_formal_20260617.json`
4. active `strategy_manifest.json`
5. active signal / delivery manifest

硬要求：

1. `db_path` 不得再指向 `quant_production.duckdb`
2. 不得再引用包含 `.BJ` 的上游资产
3. `audit_record` 必须对应新资产审计记录，不得继续挂旧 pending 或旧单库文件

### 第四优先级：active path

以下默认主线路径必须一起改：

1. `quant/data_file/production_assets/duckdb/quant_production.duckdb`
   - 降级为历史单库观察资产，不再作为新主线路径
2. 新主线路径根目录：
   - `quant/data_file/production_assets/duckdb/production/`
3. 各层各表独立文件必须进入各自目录：
   - `production/l1/`
   - `production/l2/`
   - `production/l3/`
   - `production/l4/`
   - `production/l5/`
   - `production/l6/`
   - `production/l7/`

## 过渡态与终态资产语义

### 过渡态

过渡态仅指迁移执行期间，用于保证迁移可审计、可比对、可回看，但不代表长期目标架构。

过渡态允许临时保留以下对象：

1. 旧 SQLite 资产
2. 旧 Parquet 资产
3. 旧单文件 `quant_production.duckdb`
4. 旧 split DB
5. 与旧资产绑定的 change / audit / handoff 证据文件

过渡态限制：

1. 这些资产只能作为迁移中遗留、审计证据或短期回看对象
2. 不得在新的 production route、current registry、active manifest、发布说明中继续作为正常主线组件
3. 不得再新增依赖这些旧资产的新主线脚本或新主线表

### 终态

终态是本次规则变更完成后的唯一目标架构。

终态硬规则：

1. 生产链只允许 DuckDB 一种存储形态
2. DuckDB 只允许单表单文件，不允许回到单文件多表
3. current route / active registry / active manifest / 发布说明中都不再保留 SQLite 作为正常 rollback 组件
4. 旧 SQLite、旧 Parquet、旧单文件 DuckDB、旧 mixed-db 资产都必须删除、归档或转入纯证据区，不得继续留在生产主线语义中
5. 生产链稳定后，多余中间文件也必须清理，不得长期堆积在主数据目录
6. 前复权值及其派生字段在终态必须显式带 `qfq` 语义，不允许继续依赖裸字段隐含口径

## 旧 SQLite / Parquet / 单文件 DuckDB 资产降级标准

### 过渡态立即降级为临时遗留 / 审计证据的资产

#### L1

1. `quant/data_file/raw_table_dbs/[table].DB`
   - 过渡态可临时保留为迁移遗留
2. `quant/data_file/odb.db`
   - 直接定性为 `evidence_only`

#### L2

1. `quant/data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA`
   - 过渡态可临时保留为迁移遗留
2. `odb.db.STOCK_DAILY_DATA`
   - 定性为 `evidence_only`

#### L3

1. `quant/data_file/production_factor_parts/`
   - 过渡态可临时保留为迁移遗留
2. `quant/data_file/prediction_label_parts/`
   - 过渡态可临时保留为迁移遗留
3. `quant/data_file/stock_factor_data.parquet`
   - 定性为 `evidence_only`

#### L4

1. `quant/data_file/model_predictions/MODEL_PREDICTIONS.db`
   - 过渡态可临时保留为迁移遗留
2. `odb.db.stock_predict_data_*`
   - 定性为 `evidence_only`

#### L1-L7 单库 DuckDB

1. `quant/data_file/production_assets/duckdb/quant_production.duckdb`
   - 过渡态定性为迁移观察资产或 `evidence_only`
   - 不再允许新增主线表
   - 不再允许作为新表的默认物化目标

### 过渡态降级判定标准

资产只有在同时满足以下条件时，才能从 active 降级并退出主线：

1. 对应新单表单文件 DuckDB 已通过审计
2. route 已切到新资产
3. registry 已写入 `superseded_by`
4. manifest 已不再引用旧资产
5. rollback 观察期和回滚入口已明确

### 终态清理标准

以下条件同时满足后，应进入终态清理：

1. 新 DuckDB 单表单文件主线已稳定运行
2. 相关层至少完成一轮正式增量或正式消费验证
3. 审计确认旧资产不再被 current route / registry / manifest 引用
4. 旧资产所需证据已转入审计报告目录或专用 archive / evidence 区

进入终态清理后：

1. 旧 SQLite 生产资产应删除或迁出主数据目录
2. 旧 Parquet 主线资产应删除或迁出主数据目录
3. 旧单文件 `quant_production.duckdb` 应删除或迁出主数据目录
4. 多余中间文件应删除，不得继续占用生产资产目录
5. 若确需保留证据，必须进入独立 `evidence` / `archive` 区，且 `allowed_for_main_workflow=false`

## L2 / L3 / L4 最小闭环路径

### L2 最小闭环

1. L1 新规则先落地：
   - 源头过滤 `.BJ`
   - L1 新单表单文件 DuckDB 生成并通过审计
2. 用新 L1 做一次 L2 全量重建
3. 生成：
   - `production/l2/STOCK_DAILY_DATA.duckdb::STOCK_DAILY_DATA`
4. 审计要求：
   - `.BJ=0`
   - 最新交易日行数与新 L1 驱动 universe 一致
   - write bypass 清零
   - 若价格实际为前复权，则字段命名或 schema/manifest 约束必须显式反映 `qfq`
5. 审计通过后：
   - 切 L2 active route
   - 再切 L2 registry

### L3 最小闭环

1. L2 新正式资产已 active
2. 基于新 L2 全量重建：
   - production features
   - training labels
3. 生成：
   - `production/l3/<feature_table>.duckdb`
   - `production/l3/<label_table>.duckdb`
4. 审计要求：
   - `.BJ=0`
   - feature / label 行数、键域、日期范围一致
   - Parquet / old DuckDB 不再是默认主线
   - 所有基于前复权价格派生的字段和因子都已显式声明 `qfq` 语义
5. 审计通过后：
   - 先切 L3 route
   - 再切 L3 registry

### L4 最小闭环

1. L3 新正式资产已 active
2. 基于新 L3 重新发布 formal 3D/5D/10D 预测资产
3. 生成：
   - 各自单表单文件 DuckDB
   - 对应 active formal manifest
4. 审计要求：
   - `.BJ=0`
   - manifest `db_path/table/audit_record` 全部指向新资产
   - L4 默认读路由不再回落旧单库或旧 SQLite
   - L4 input contract / manifest 已显式写清 adjustment semantics
5. 审计通过后：
   - 先切 L4 manifest / route
   - 再切 L4 registry
   - 再允许 L5/L6/L7 继续消费

## 文档跟随真实路由同步改写规则

为了避免“假落地”，文档改写必须与真实路由状态绑定，不允许提前宣布已经完成。

### 必须跟随真实路由一起改的文档

1. `D:/work/quant/quant_mcp/WORKFLOW.md`
2. `D:/work/quant/quant_mcp/quant/main/AGENTS.md`
3. `D:/work/quant/quant_mcp/quant/main/README.md`
4. `D:/work/quant/quant_mcp/quant/data_file/asset_registry/README.md`
5. `D:/work/quant/quant_mcp/quant/main/docs/governance/local-duckdb-migration-evaluation.md`
6. 各层 change / audit / handoff 说明文档

### 文档改写时点

#### 第一批：可先改“规则已批准，执行中”

可立即改写为：

1. 项目股票范围不再包含北交所
2. DuckDB 目标架构为单表单文件
3. 当前处于迁移执行期
4. 旧单库和旧 SQLite / Parquet 仅作为 rollback / evidence 保留

但不得写成：

1. “当前所有层已完成切换”
2. “当前默认生产入口已全部改为单表单文件”

#### 第二批：必须在对应 route + registry 真切换后改

1. `WORKFLOW.md`
   - 当前默认资产口径
   - 当前 active 层级路径
2. `AGENTS.md`
   - 标准资产入口
   - DuckDB 主线描述
3. `README.md`
   - Current default assets
   - Quick start 中默认链路说明
4. `asset_registry/README.md`
   - active / rollback / evidence 语义示例

### 文档收口标准

文档只有在同时满足以下条件时，才能把新路径写成“当前默认主线”：

1. 对应 route 已切换
2. 对应 registry 已切换
3. 对应 manifest 已切换
4. 审计 approved 记录已存在
5. 旧资产已被明确降级

## 迁移顺序

建议采用七步顺序，逐层止损：

1. **先冻结规则**
   - 文档和决策口径统一为“排除 `.BJ` + 单表单文件 DuckDB”
2. **先收口 active route**
   - 先把默认入口从“单库 / 含 `.BJ` 旧资产”改成“新规则待接入点”
3. **再收口 active manifest 与 registry 变更草案**
   - 形成新 change file，但暂不切 active
4. **L1 规则收口与正式重物化**
   - 修 stock pool、source probe、raw sync、DuckDB 路由
   - 生成无 `.BJ` 的 L1 新资产
5. **L2 全量重建与审计**
   - 基于新 L1 重建 `STOCK_DAILY_DATA`
6. **L3 全量重建与审计**
   - 基于新 L2 重建 features / labels
7. **L4 发布口径重建与审计**
   - 基于新 L3 发布新 formal manifests 和正式预测资产
8. **L5/L6/L7/L8 改路由、切 registry、补文档**
   - 在上游通过后切换读路由、补历史证据说明、统一外部契约

## 阻断条件

以下任一项成立，禁止进入下一层：

1. 当前层新正式资产仍包含 `.BJ`
2. 当前层 route 仍默认读取 `quant_production.duckdb`
3. 当前层 registry / manifest 仍指向旧单库资产
4. 当前层 change file / audit record 未与新资产一致
5. 当前层 rollback 语义未定义
6. 当前层 bypass scan 仍存在旧写链直接写入旧资产

## 审计点

每层至少审以下内容：

1. **股票范围审计**
   - `.BJ` 行数必须为 `0`
2. **路径审计**
   - 正式 route / manifest / registry 必须指向单表单文件 DuckDB
3. **一致性审计**
   - 新资产与上游来源行数、字段、日期范围、自定义 hash 抽样一致
4. **口径审计**
   - 值为前复权但字段名未显式标记，或 contract/manifest 未写清 adjustment semantics，直接按阻断缺陷处理
5. **写链覆盖审计**
   - 不存在绕过新路由继续写旧单库或旧 SQLite / Parquet 主线的旁路
6. **rollback 审计**
   - 旧资产保留但 `allowed_for_main_workflow=false`
7. **文档审计**
   - 不再把 `.BJ` 或 `quant_production.duckdb` 写成当前生产默认口径

## 最小测试清单

### 通用测试

1. route unit tests
2. registry gate tests
3. manifest parse / fail-closed tests
4. bypass scan tests
5. `.BJ` 零命中断言测试
6. adjustment semantics / `qfq` 命名一致性测试

### L1

1. `stock_pool_module` 排除 `.BJ` 单测
2. `run_all_a_raw_update` source preflight 单测
3. L1 DuckDB sync / rebuild 单测

### L2

1. `stock_daily_data_route` 新路径解析单测
2. target-date integration 读写契约测试
3. L2 update contract validation

### L3

1. feature / label 新 DuckDB 路由单测
2. full rebuild 和 incremental rebuild 契约测试
3. L3 bypass scan

### L4

1. formal manifest 指向新 DuckDB 文件测试
2. model route 解析测试
3. light-factor / rolling-train 分支不得回退旧主线测试

### L5 / L6 / L7

1. strategy route / signal route 新路径解析测试
2. bundle contract validation
3. 旧目录型资产仅在显式 rollback 场景可读测试

## 责任归口

- `architect-agent`
  - 统一新路径标准、registry 语义、manifest 契约、迁移顺序和审计门禁
- `audit-agent`
  - 对每层执行只读复核，重点拦截 `.BJ` 残留、旧单库误用和 rollback 语义缺失
- `data-ingestion-agent`
  - L1 股票池、源头过滤、raw sync、L1 DuckDB 拆分
- `data-integration-agent`
  - L2 全量重建、L2 单表单文件 DuckDB 主线
- `factor-agent`
  - L3 全量重建、L3 正式资产拆分与旁路清零
- `model-agent`
  - L4 新正式输入路由、formal manifest、预测资产重发布
- `strategy-agent`
  - L5/L6 路由、bundle 契约、历史说明收口
- `trading-agent`
  - L7 signal 读路由和交付说明收口
- `mcp-agent`
  - L8 发布层文档和外部资产网关契约同步

## 本轮建议

1. 本轮只批准进入“方案固化 + 分层整改设计”阶段
2. 不批准任何层直接删数据
3. 不批准任何层直接切 production registry
4. 先由 `data-ingestion-agent` 启动 L1 规则整改设计，再由 `audit-agent` 复核
5. L2-L8 必须等待上游新正式资产通过审计后再进入各自执行阶段

## 是否需要用户审批

需要。

原因：

1. 这是项目级股票范围变更
2. 这是生产 DuckDB 存储布局变更
3. 后续会触发逐层全量重建和逐层正式资产切换，必须保持主人审批和审计闭环

## 边界声明

本次仅完成只读评审与治理方案输出：

1. 未执行业务脚本
2. 未删除数据
3. 未修改 production registry
4. 未触发训练、预测、信号或交易
