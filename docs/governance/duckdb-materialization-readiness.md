> 更新（2026-07-02）：本文件保留送审准备模板语义；当前 production current route 已切到 split DuckDB-only。文中若出现旧 SQLite / Parquet / shared-DuckDB，只能按历史迁移或证据留痕理解。
# DuckDB 真实物化前准备规范

## 目标

本文档定义 DuckDB 主线改造进入“真实按层物化前准备”阶段时，必须提前准备的证据、边界和送审口径。

本阶段只允许做准备，不允许：
- 执行真实物化
- 写入 `production_assets.json`
- 把任何 DuckDB 资产标记为 `production_active`
- 切换默认主线读取入口

## 适用范围

仅覆盖 `production_assets.json` 当前登记且 `allowed_for_main_workflow=true` 的 L1-L7 正式资产。

不纳入本阶段准备范围：
- `history/`
- `smoke/`
- `backup/`
- `reports/`
- `research/`
- legacy 复现实验资产

## 阶段目标

进入真实物化前，至少要先把下面四类内容准备齐：

1. 源资产清单
2. 分层落点清单
3. 表级一致性校验口径
4. 审计送审包

## 责任归口

### 架构师智能体

负责：
- 统一阶段边界
- 统一送审口径
- 统一目录和命名规范
- 检查默认主线是否仍保持在当前正式资产

不负责：
- 真实物化执行
- 生产资产写入
- 数据补齐

### 数据整合智能体

负责：
- 组织 L1-L3 源资产到 DuckDB 目标层的映射清单
- 准备表级一致性校验结果
- 准备重复键、日期范围、字段数、样本 hash 等证据

不负责：
- 擅自切主线
- 擅自写 registry

### 审计智能体

负责：
- 对准备包做只读复核
- 判断是否允许进入真实物化
- 在真实物化前继续做门禁审计

## 准备包目录规范

建议每次准备包都单独放在：

```text
quant/data_file/reports/duckdb_migration_YYYYMMDD_materialization_prep/
```

目录内至少包含：

```text
duckdb_materialization_readiness.md
duckdb_materialization_readiness_manifest.json
source_asset_inventory.csv
table_consistency_checks.csv
rollback_mapping.csv
```

## 必备证据

### 1. 源资产清单

每个待迁移资产至少要列出：
- `asset_id`
- `layer`
- `owner_agent`
- `asset_type`
- `asset_path`
- `source_audit_record`
- `allowed_for_main_workflow`

### 2. DuckDB 目标落点

当前主线必须明确为 split DuckDB 目标文件，而不是单文件多表：
- `L1 -> quant/data_file/production_assets/duckdb/l1_raw_tables/[table].duckdb`
- `L2 -> quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb`
- `L3 feature -> quant/data_file/production_assets/duckdb/l3_feature_current.duckdb`
- `L3 label -> quant/data_file/production_assets/duckdb/l3_label_current.duckdb`
- `L4 -> quant/data_file/production_assets/duckdb/l4_*.duckdb`
- `L5-L7 current -> quant/data_file/production_assets/duckdb/production/l5|l6|l7/*.duckdb`

### 3. 表级一致性校验

每张表至少要给出：
- 源路径
- 目标 DuckDB 文件
- 目标表名
- 行数
- 字段数
- 日期范围
- 重复键检查结果
- 样本 hash
- 是否通过

如果某张表没有明确主键或日期字段，也必须在准备包里写明原因，不能跳过不写。

### 4. rollback 映射

每个 DuckDB 目标资产必须显式登记 rollback 来源：
- 对应 SQLite / Parquet 源资产
- rollback 用途
- rollback 是否可直接只读打开

## 送审前硬检查

进入真实物化送审前，至少要确认：

1. 默认主线读取已经指向 split DuckDB 正式资产
2. DuckDB 仅通过显式后端选择进入
3. `production_assets.json` 未新增 DuckDB 正式资产
4. `quant/data_file/production_assets/duckdb/` 未出现被误认作生产切换的残留产物，或已明确标注其用途
5. 每个源资产都已有审计记录
6. `duckdb_migration_gate.py` 和 `production_asset_gate.py` 的 fail-closed 规则未被放松

## 审计回执最少字段

送审给审计智能体时，回执至少应能回答：
- 是否允许进入真实物化
- 风险等级
- 哪些层允许继续
- 哪些层仍缺证据
- 是否需要主人审批

## 模板位置

标准模板放在：

```text
quant/data_file/reports/templates/duckdb_materialization_readiness/
```

下次进入真实物化前准备阶段，优先复制模板再填证据，不要现场临时拼结构。
