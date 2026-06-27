# 远程数仓 MCP 中台部署包

本目录用于在独立服务器上部署量化远程数仓基础设施。

当前部署包覆盖：

- ClickHouse：承载 L1-L7 业务大表和分析查询。
- PostgreSQL：承载 registry、manifest、lineage、audit、ops 等治理元数据。
- MinIO：承载 Parquet 快照、模型文件、报告、回测归档、交易归档和备份。
- MCP skeleton：启动 `asset-registry-mcp`、`ops-mcp`、`audit-mcp` 的健康检查和元数据服务骨架。

MCP 子服务在本阶段只包含健康检查、服务元数据和计划工具清单，尚未包含真实数据库读写工具。不要把当前部署包理解为已经完成数据迁移、生产入口切换或正式 MCP 资产服务上线。

## 目录结构

```text
deploy/cloud-center/
  README.md
  docker-compose.yml
  env.example
  sql/
    clickhouse_init.sql
    postgres_init.sql
  mcp/
    Dockerfile
    cloud_center_mcp/
  scripts/
    health_check.sh
    health_check.ps1
```

## 服务器目录

建议服务器使用：

```text
/opt/quant_mcp/
  repo/
  env/
  volumes/
    clickhouse/
    postgres/
    minio/
    mcp_logs/
  backups/
  runtime/
```

## 初始化步骤

```bash
sudo mkdir -p /opt/quant_mcp/{env,volumes/clickhouse,volumes/postgres,volumes/minio,volumes/mcp_logs,backups,runtime}
sudo chown -R "$USER":"$USER" /opt/quant_mcp

cd /opt/quant_mcp
git clone https://github.com/wangjrking/quant_vibe.git repo
cd repo
git checkout cloud-center-mcp

cp deploy/cloud-center/env.example /opt/quant_mcp/env/warehouse.env
```

编辑 `/opt/quant_mcp/env/warehouse.env`，填入本机密钥。密钥不得提交到 Git。

启动：

```bash
cd /opt/quant_mcp/repo
docker compose --env-file /opt/quant_mcp/env/warehouse.env -f deploy/cloud-center/docker-compose.yml up -d
```

健康检查：

```bash
bash deploy/cloud-center/scripts/health_check.sh
```

Windows PowerShell 远程或本机检查：

```powershell
.\deploy\cloud-center\scripts\health_check.ps1
```

Win10 宿主机使用单独的 Compose 文件和健康检查脚本：

```powershell
docker context use desktop-linux
docker compose --env-file D:\quant\cloud-center\env\warehouse.env -f D:\quant\cloud-center\repo\deploy\cloud-center\docker-compose.win10.yml up -d
D:\quant\cloud-center\repo\deploy\cloud-center\scripts\health_check_win10.ps1
```

Win10 版本仅适用于当前阶段远程中台试运行。生产资产迁移、主线默认入口切换和自动同步任务仍需单独审批。

## 默认端口

| 服务 | 宿主机端口 | 容器端口 | 说明 |
| --- | ---: | ---: | --- |
| ClickHouse HTTP | 8123 | 8123 | SQL HTTP 接口 |
| ClickHouse Native | 9000 | 9000 | Native TCP 接口 |
| PostgreSQL | 5432 | 5432 | 治理元数据库 |
| MinIO API | 9100 | 9000 | 对象存储 API |
| MinIO Console | 9101 | 9001 | 对象存储控制台 |
| asset-registry-mcp | 8101 | 8101 | MCP 骨架服务 |
| ops-mcp | 8102 | 8102 | MCP 骨架服务 |
| audit-mcp | 8103 | 8103 | MCP 骨架服务 |

## 验收标准

- ClickHouse `SELECT 1` 返回成功。
- PostgreSQL `pg_isready` 返回 accepting connections。
- MinIO health live 返回成功。
- 三个 MCP skeleton 服务 `/health` 返回成功。
- ClickHouse 存在 `l1_raw`、`l2_base`、`l3_feature`、`l4_model`、`l5_strategy`、`l6_backtest`、`l7_trading`。
- PostgreSQL 存在 `registry`、`audit`、`ops` schema。

## 边界

- 本部署包不迁移本地业务数据。
- 本部署包不切换主线默认读取入口。
- 本部署包不启动训练、预测、信号、回测或交易。
- 本部署包的 MCP skeleton 不提供生产数据读写工具。
- 生产资产迁移前必须先由审计智能体只读复核并留存记录。
