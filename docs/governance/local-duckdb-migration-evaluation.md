> 更新（2026-07-02）：当前 active production 已完成 no-BJ + split DuckDB-only 切换。本文保留迁移评估背景；文中出现的 SQLite / Parquet / `quant_production.duckdb` 仅代表历史迁移阶段或待清理遗留，不代表当前默认主链。
# 本地 DuckDB 主线改造方案

## 目标

本文档定义本项目从本地 `SQLite + Parquet` 资产形态迁移到 DuckDB 主线资产形态的改造方案。

当前实现口径：

- 只覆盖 `production_assets.json` 登记的 L1-L7 正式资产。
- 历史、smoke、backup、reports、research 目录不进入 DuckDB 主线。
- DuckDB 按层拆分为多个文件，不使用单一巨型库。
- 历史迁移阶段曾允许保留 SQLite / Parquet 作为 legacy rollback；按当前终态治理口径，它们不再属于 current production component。
- DuckDB 资产必须完成迁移校验和审计门禁后，才允许进入生产 registry 和主线默认读取。

## 当前结论

DuckDB-only 已成为当前主线目标；本文件保留的是迁移评估与门禁背景，不代表 today current route 仍停留在旧资产。

历史迁移阶段路线：

```text
阶段 1：只读评估和 dry-run
阶段 2：生成按层 DuckDB 正式资产镜像
阶段 3：迁移一致性报告和审计门禁
阶段 4：审计通过后更新 production_assets registry
阶段 5：主线读取切换到 DuckDB，旧 SQLite / Parquet 退出 current route
```

在未完成迁移审计前，不得把 DuckDB 产物标记为 production active。

## 目标 DuckDB 文件

```text
quant/data_file/production_assets/duckdb/
  l1_raw_tables/[table].duckdb
  l2_stock_daily_data.duckdb
  l3_feature_current.duckdb
  l3_label_current.duckdb
  l4_*.duckdb
  production/l5/*.duckdb
  production/l6/*.duckdb
  production/l7/*.duckdb
```

## 已实现工具

```text
quant/main/duckdb_asset_route.py
quant/main/tools/materialize_duckdb_production_assets.py
quant/main/tools/duckdb_migration_gate.py
```

工具边界：

- `duckdb_asset_route.py` 统一定义 DuckDB 路径、后端选择、只读连接和 legacy rollback opt-in。
- `materialize_duckdb_production_assets.py` 只读取 production registry，产出 DuckDB 文件和迁移报告，不改 registry。
- `duckdb_migration_gate.py` 对迁移报告、源资产审计记录、审批审计记录和 DuckDB 文件执行 fail-closed 校验。
- `production_asset_gate.py` 已补 DuckDB 专项检查，没有 `duckdb_migration_manifest`、`rollback_source_asset` 和审计记录时不得进入 production active。

## 主线读取规则

- L2 `STOCK_DAILY_DATA` 路由支持 `duckdb` 和 `legacy` 两种后端。
- 在 DuckDB 正式资产完成物化、审计通过并写入 production registry 之前，当前默认后端应为 split DuckDB 主线；旧 SQLite / Parquet 只允许作为历史证据或归档读取。
- DuckDB 只能通过显式后端选择进入；显式 legacy rollback 读取必须设置 `QUANT_ALLOW_LEGACY_SQLITE_PARQUET_ROLLBACK=1`。
- L3 features、L3 labels、L4 predictions 已提供 DuckDB 路径解析函数。
- L4 formal manifest 支持 `source_type=duckdb_table`。
- DuckDB prediction manifest 必须包含 `audit_record` 和 `duckdb_migration_manifest`，否则拒绝生产读取。

## 审计门槛

只有同时满足以下条件，才允许把 DuckDB 资产写入 production registry 并切换主线：

1. DuckDB 物化报告存在。
2. 每层 DuckDB 文件存在。
3. 每个来源资产已有审计记录。
4. 每个目标表行数、字段和样本 hash 校验通过。
5. 审计智能体出具 DuckDB 迁移审批记录。
6. `duckdb_migration_gate.py` 通过。
7. `production_asset_gate.py` 通过。

## 真实物化前准备

在进入真实按层物化前，先按下列规范组织准备包：

- 规范文档：`quant/main/docs/governance/duckdb-materialization-readiness.md`
- 模板目录：`quant/data_file/reports/templates/duckdb_materialization_readiness/`

本阶段仍然不允许直接执行真实物化，不允许写 registry，不允许切主线。

## 禁止事项

- 不把旧 SQLite、Parquet 或 shared-DuckDB 重新接回 current production route。
- 不直接修改 `production_assets.json` 伪造主线切换。
- 不伪造审计记录。
- 不跑因子、模型、策略、信号或回测。
- 不把未审计 DuckDB 产物标记为生产资产。
- 不把实验 DuckDB 产物设为主线默认读取入口。

## 运行示例

dry-run：

```powershell
D:/work/quant/quant_mcp/.venv/Scripts/python.exe D:/work/quant/quant_mcp/quant/main/tools/materialize_duckdb_production_assets.py --dry-run
```

按层物化示例：

```powershell
D:/work/quant/quant_mcp/.venv/Scripts/python.exe D:/work/quant/quant_mcp/quant/main/tools/materialize_duckdb_production_assets.py --layer L2
```

门禁示例：

```powershell
D:/work/quant/quant_mcp/.venv/Scripts/python.exe D:/work/quant/quant_mcp/quant/main/tools/duckdb_migration_gate.py --report quant/data_file/reports/duckdb_migration_YYYYMMDD/duckdb_migration_report.json --approval-audit-record quant/data_file/reports/<audit>.md
```
