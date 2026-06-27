# 远程数仓 MCP 中台部署包

本目录用于在独立服务器上部署量化远程数仓基础设施。

当前部署包覆盖：

- ClickHouse：承载 L1-L7 业务大表和分析查询。
- PostgreSQL：承载 registry、manifest、lineage、audit、ops 等治理元数据。
- MinIO：承载 Parquet 快照、模型文件、报告、回测归档、交易归档和备份。

MCP 子服务在本阶段只定义端口、权限和服务边界，尚未包含真实 MCP 服务实现。不要把当前部署包理解为已经完成数据迁移或生产入口切换。

## 目录结构

```text
deploy/cloud-center/
  README.md
  docker-compose.yml
  env.example
  sql/
    clickhouse_init.sql
    postgres_init.sql
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

## 默认端口

| 服务 | 宿主机端口 | 容器端口 | 说明 |
| --- | ---: | ---: | --- |
| ClickHouse HTTP | 8123 | 8123 | SQL HTTP 接口 |
| ClickHouse Native | 9000 | 9000 | Native TCP 接口 |
| PostgreSQL | 5432 | 5432 | 治理元数据库 |
| MinIO API | 9100 | 9000 | 对象存储 API |
| MinIO Console | 9101 | 9001 | 对象存储控制台 |

## 验收标准

- ClickHouse `SELECT 1` 返回成功。
- PostgreSQL `pg_isready` 返回 accepting connections。
- MinIO health live 返回成功。
- ClickHouse 存在 `l1_raw`、`l2_base`、`l3_feature`、`l4_model`、`l5_strategy`、`l6_backtest`、`l7_trading`。
- PostgreSQL 存在 `registry`、`audit`、`ops` schema。

## 边界

- 本部署包不迁移本地业务数据。
- 本部署包不切换主线默认读取入口。
- 本部署包不启动训练、预测、信号、回测或交易。
- 生产资产迁移前必须先由审计智能体只读复核并留存记录。
