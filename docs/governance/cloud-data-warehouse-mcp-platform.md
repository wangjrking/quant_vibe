# 远程数仓 MCP 中台方案

## 目标

本文档定义本项目的远程数仓 MCP 中台方案。目标是在一台独立服务器上部署 ClickHouse、PostgreSQL、对象存储和多个 MCP 子服务，把数据、因子、模型结果、策略归档和交易交付证据统一放到云端资产层，并通过受控 MCP 接口对各智能体提供服务。

本文档面向另一台服务器执行。服务器拉取本仓库后，可按本文档完成基础部署、目录初始化、schema 初始化、服务拆分和验收。

## 总体结论

推荐架构为：

```text
ClickHouse + PostgreSQL + 对象存储 + 多 MCP 子服务
```

- ClickHouse 负责业务大表和分析查询，包括 L1 原始表镜像、L2 综合底表、L3 因子标签、L4 预测结果、L5 信号明细、L6 回测明细和 L7 交易交付明细。
- PostgreSQL 负责治理元数据，包括资产 registry、manifest、lineage、任务状态、审计记录、审批状态、权限配置和工作流状态。
- 对象存储负责 Parquet、模型文件、报告、回测归档、信号归档、快照和大体量历史资产。
- MCP 子服务按职责拆分，分别提供资产查询、数据读取、因子读取、模型结果读取、策略归档查询、交易交付查询和审计证据查询。智能体默认通过 MCP 调用，不直接连接数据库。

不建议把全部数据都塞进单一 PostgreSQL 表。日线、因子、预测和信号这类高行数、宽表、按日期和股票批量扫描的数据，应进入 ClickHouse。PostgreSQL 保持治理元数据库定位。SQLite 只适合本地缓存和开发验证，不适合远程多智能体并发读写。

## 目标服务器假设

最低建议配置：

```text
CPU：8 核以上
内存：32 GB 以上
系统盘：200 GB SSD
数据盘：2 TB 以上，优先 SSD 或高可靠云盘
系统：Ubuntu Server 22.04 LTS 或 24.04 LTS
网络：内网固定 IP，必要时通过 VPN / ZeroTier / Tailscale 访问
部署方式：Docker Compose
```

生产建议：

```text
CPU：16 核以上
内存：64 GB 以上
数据盘：4 TB 以上，启用快照
备份盘或远程备份：至少保留 30 天
```

## 服务组件

| 组件 | 推荐技术 | 职责 |
| --- | --- | --- |
| 分析数据库 | ClickHouse 24+ | L1-L7 业务大表、按交易日/股票/因子/评分的高速查询 |
| 治理数据库 | PostgreSQL 16+ | registry、manifest、lineage、审计、任务状态、权限和审批状态 |
| 对象存储 | MinIO 或 S3 兼容服务 | Parquet、模型文件、报告、信号、回测归档、快照 |
| MCP 网关 | Python / FastMCP | 对外提供受控工具接口 |
| 任务状态库 | PostgreSQL schema `ops` | workflow monitor、执行状态、锁和审计闭环 |
| 监控 | Prometheus + Grafana，可先文档化 | ClickHouse、PostgreSQL、对象存储、MCP 健康状态 |
| 备份 | ClickHouse backup + pg_dump + 对象存储版本化 | 数据恢复和灾备 |

## 数据分层

云端分层应保持本地 L1-L7 语义，不重新定义业务层级。

| 层级 | 中文名称 | ClickHouse database | PostgreSQL schema | 对象存储 prefix | 默认用途 |
| --- | --- | --- | --- | --- | --- |
| L1 | 原始数据层 | `l1_raw` | `registry` | `s3://quant-assets/l1_raw/` | Tushare 等源头原始表镜像、原始分区索引 |
| L2 | 综合底表层 | `l2_base` | `registry` | `s3://quant-assets/l2_base/` | `STOCK_DAILY_DATA` 与派生底表快照 |
| L3 | 因子标签层 | `l3_feature` | `registry` | `s3://quant-assets/l3_feature/` | 生产因子、实验因子、标签分区 |
| L4 | 模型结果层 | `l4_model` | `registry` | `s3://quant-assets/l4_model/` | 模型评分、预测结果、formal manifest |
| L5 | 策略信号层 | `l5_strategy` | `registry` | `s3://quant-assets/l5_strategy/` | 策略规则、候选信号、正式信号资产 |
| L6 | 回测验证层 | `l6_backtest` | `registry` | `s3://quant-assets/l6_backtest/` | 本地和掘金回测明细、策略验收记录 |
| L7 | 交易交付层 | `l7_trading` | `registry` | `s3://quant-assets/l7_trading/` | 掘金/QMT 交付、模拟盘/实盘执行证据 |
| OPS | 治理运行层 | 不放业务大表 | `ops` / `audit` / `registry` | `s3://quant-assets/ops/` | 任务状态、审计记录、监控日报 |

## 生产资产与实验资产

生产资产和实验资产必须物理隔离、权限隔离、manifest 隔离。

推荐对象存储路径：

```text
s3://quant-assets/production/l1/
s3://quant-assets/production/l2/
s3://quant-assets/production/l3/
s3://quant-assets/production/l4/
s3://quant-assets/production/l5/
s3://quant-assets/production/l6/
s3://quant-assets/production/l7/

s3://quant-assets/experimental/l1/
s3://quant-assets/experimental/l2/
s3://quant-assets/experimental/l3/
s3://quant-assets/experimental/l4/
s3://quant-assets/experimental/l5/
s3://quant-assets/experimental/l6/
s3://quant-assets/experimental/l7/
```

规则：

- 主线工作流只能读取 `production` 轨道。
- 实验联调只能读取 `experimental` 轨道。
- 实验资产不得通过“最新文件”隐式进入主线。
- L4 到 L5 必须使用 approved formal manifest。
- 正式资产变动前必须先由审计智能体出具审计记录。

## ClickHouse 业务库设计

ClickHouse 是本方案的业务分析数据库，负责承载需要高频查询、按交易日批量扫描、按股票或因子聚合排序的大表。

推荐 database：

```sql
CREATE DATABASE IF NOT EXISTS l1_raw;
CREATE DATABASE IF NOT EXISTS l2_base;
CREATE DATABASE IF NOT EXISTS l3_feature;
CREATE DATABASE IF NOT EXISTS l4_model;
CREATE DATABASE IF NOT EXISTS l5_strategy;
CREATE DATABASE IF NOT EXISTS l6_backtest;
CREATE DATABASE IF NOT EXISTS l7_trading;
```

典型表设计原则：

- 分区键优先使用交易日月份，例如 `toYYYYMM(trade_date)`。
- 排序键优先使用 `(trade_date, ts_code)` 或 `(strategy_id, signal_date, ts_code)`。
- 生产轨道和实验轨道必须显式字段区分，推荐字段为 `track`。
- 所有可进入主线的业务表必须能追溯到 PostgreSQL `registry.assets.asset_id`。
- 大文件和原始快照仍放对象存储；ClickHouse 保存可查询明细和必要索引。

L2 综合底表示例：

```sql
CREATE TABLE IF NOT EXISTS l2_base.stock_daily_data
(
    trade_date Date,
    ts_code String,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    vol Float64,
    amount Float64,
    adj_factor Nullable(Float64),
    st_type Nullable(String),
    asset_id String,
    track LowCardinality(String),
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, ts_code);
```

L3 因子表示例：

```sql
CREATE TABLE IF NOT EXISTS l3_feature.factor_values
(
    trade_date Date,
    ts_code String,
    factor_name LowCardinality(String),
    factor_value Nullable(Float64),
    asset_id String,
    track LowCardinality(String),
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, factor_name, ts_code);
```

L4 预测结果表示例：

```sql
CREATE TABLE IF NOT EXISTS l4_model.predictions
(
    trade_date Date,
    ts_code String,
    horizon LowCardinality(String),
    score Float64,
    rank UInt32,
    model_id String,
    asset_id String,
    manifest_id String,
    track LowCardinality(String),
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, horizon, score, ts_code);
```

## PostgreSQL 治理库设计

PostgreSQL 是本方案的治理元数据库，不承载大体量业务明细。它负责回答“这个资产是谁生产的、是否审计通过、能否被下游主线读取、源头和去向是什么”。

### registry schema

建议建立 `registry` schema，管理所有云端资产索引。

核心表：

```sql
CREATE SCHEMA IF NOT EXISTS registry;

CREATE TABLE IF NOT EXISTS registry.assets (
    asset_id text PRIMARY KEY,
    asset_name text NOT NULL,
    layer text NOT NULL,
    track text NOT NULL CHECK (track IN ('production', 'experimental', 'legacy')),
    owner_agent text NOT NULL,
    storage_type text NOT NULL CHECK (storage_type IN ('clickhouse', 'object_storage', 'postgres', 'external')),
    storage_uri text NOT NULL,
    manifest_uri text,
    database_name text,
    table_name text,
    version text NOT NULL,
    trade_date text,
    status text NOT NULL,
    audit_record_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS registry.manifests (
    manifest_id text PRIMARY KEY,
    layer text NOT NULL,
    track text NOT NULL,
    owner_agent text NOT NULL,
    manifest_uri text NOT NULL,
    approved_for_mainline boolean NOT NULL DEFAULT false,
    approved_for_l5 boolean NOT NULL DEFAULT false,
    audit_record_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS registry.lineage (
    lineage_id bigserial PRIMARY KEY,
    output_asset_id text NOT NULL,
    input_asset_id text NOT NULL,
    relation_type text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
```

### audit schema

```sql
CREATE SCHEMA IF NOT EXISTS audit;

CREATE TABLE IF NOT EXISTS audit.records (
    audit_record_id text PRIMARY KEY,
    request_agent text NOT NULL,
    audit_agent text NOT NULL DEFAULT 'audit-agent',
    layer text NOT NULL,
    asset_id text,
    conclusion text NOT NULL,
    risk_level text NOT NULL,
    evidence_uri text NOT NULL,
    approved_for_production boolean NOT NULL DEFAULT false,
    approved_for_downstream boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);
```

### ops schema

```sql
CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS ops.workflow_tasks (
    task_id text PRIMARY KEY,
    workflow_name text NOT NULL,
    owner_agent text NOT NULL,
    status text NOT NULL,
    started_at timestamptz,
    eta_minutes integer,
    completed_at timestamptz,
    blocker text,
    audit_required boolean NOT NULL DEFAULT false,
    audit_closed boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
```

业务明细不得默认进入 PostgreSQL：

- 元数据、索引、manifest、状态、审计记录进入 PostgreSQL。
- 大体量日线、因子矩阵、预测分区、信号明细和回测明细进入 ClickHouse。
- Parquet、模型文件、报告和归档快照进入对象存储。
- 常用小结果表如需进入 PostgreSQL，必须明确为治理视图或状态表，并有资产 registry 记录。

## MCP 子服务拆分

不建议做一个万能 MCP 服务。推荐拆为多个职责清晰的子服务。

| 服务 | 职责 | 默认权限 |
| --- | --- | --- |
| `asset-registry-mcp` | 查询 PostgreSQL 中的资产、manifest、lineage、审计状态 | 全智能体只读，数仓/审计受控写 |
| `l1-data-mcp` | 查询 ClickHouse L1 原始表分区和覆盖情况 | 数据接入写，其他只读 |
| `l2-base-mcp` | 查询 ClickHouse L2 综合底表和覆盖状态 | 数据整合写，因子只读 |
| `l3-feature-mcp` | 查询 ClickHouse L3 因子、标签和对应 manifest | 因子写，模型只读 |
| `l4-model-mcp` | 查询 ClickHouse L4 预测明细、对象存储模型文件和 PostgreSQL formal manifest | 模型写，策略只读 |
| `l5-strategy-mcp` | 查询 ClickHouse L5 信号、策略归档和正式信号资产 | 策略写，交易只读 |
| `l7-trading-mcp` | 查询 ClickHouse L7 交易交付和对象存储执行证据 | 交易写，审计只读 |
| `audit-mcp` | 查询 PostgreSQL 审计记录和对象存储证据路径 | 审计写，其他只读 |
| `ops-mcp` | 查询 PostgreSQL 任务状态、监控、锁和日报 | 指挥官写，其他只读 |

服务实现可以先用同一个代码仓库和同一个部署镜像，通过不同配置启动多个服务实例；不要让一个工具同时拥有跨层写权限。

## 权限模型

推荐 PostgreSQL 角色：

```text
quant_admin：管理员，仅部署和应急使用
quant_commander_rw：写 ops，读 registry/audit
quant_audit_rw：写 audit，读所有 registry 和资产索引
quant_warehouse_rw：写 registry、ops 运维表，不能写业务生产数据
quant_registry_rw：写 registry
quant_ops_rw：写 ops
quant_readonly：跨层只读，供审计和查询
```

推荐 ClickHouse 用户：

```text
ch_admin：管理员，仅部署和应急使用
ch_l1_rw：写 l1_raw
ch_l2_rw：写 l2_base
ch_l3_rw：写 l3_feature
ch_l4_rw：写 l4_model
ch_l5_rw：写 l5_strategy / l6_backtest
ch_l7_rw：写 l7_trading
ch_readonly：跨层只读，供审计和查询
```

原则：

- 每个专业智能体只获得本层写权限和必要下游/上游只读权限。
- 审计智能体默认只读业务资产，只能写审计记录。
- 数仓智能体可写 PostgreSQL `registry` 和 `ops` 运维表，不直接写 ClickHouse 业务生产数据。
- 生产资产 database / schema 不允许公共写权限。

## 部署目录

建议另一台服务器使用：

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

仓库拉取：

```bash
sudo mkdir -p /opt/quant_mcp
sudo chown -R "$USER":"$USER" /opt/quant_mcp
cd /opt/quant_mcp
git clone https://github.com/wangjrking/quant_vibe.git repo
cd repo
git checkout cloud-center-mcp
```

## 环境变量模板

在服务器创建 `/opt/quant_mcp/env/warehouse.env`：

```bash
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=quant_warehouse
POSTGRES_USER=quant_admin
POSTGRES_PASSWORD=<由主人在服务器本地填写>

CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=8123
CLICKHOUSE_NATIVE_PORT=9000
CLICKHOUSE_USER=ch_admin
CLICKHOUSE_PASSWORD=<由主人在服务器本地填写>

MINIO_ENDPOINT=http://minio:9000
MINIO_ACCESS_KEY=<由主人在服务器本地填写>
MINIO_SECRET_KEY=<由主人在服务器本地填写>
MINIO_BUCKET=quant-assets

MCP_BIND_HOST=0.0.0.0
MCP_AUTH_TOKEN=<由主人在服务器本地填写>
```

密钥不得提交到 Git。

## Docker Compose 参考结构

建议后续在仓库中补充 `deploy/cloud-center/docker-compose.yml`。第一阶段可按以下结构落地：

```yaml
services:
  clickhouse:
    image: clickhouse/clickhouse-server:24
    env_file:
      - /opt/quant_mcp/env/warehouse.env
    volumes:
      - /opt/quant_mcp/volumes/clickhouse:/var/lib/clickhouse
    ports:
      - "8123:8123"
      - "9000:9000"

  postgres:
    image: postgres:16
    env_file:
      - /opt/quant_mcp/env/warehouse.env
    volumes:
      - /opt/quant_mcp/volumes/postgres:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    env_file:
      - /opt/quant_mcp/env/warehouse.env
    volumes:
      - /opt/quant_mcp/volumes/minio:/data
    ports:
      - "9100:9000"
      - "9101:9001"

  asset-registry-mcp:
    build:
      context: .
      dockerfile: deploy/cloud-center/mcp.Dockerfile
    env_file:
      - /opt/quant_mcp/env/warehouse.env
    command: ["python", "-m", "cloud_center_mcp.asset_registry"]
    ports:
      - "8101:8101"
```

## 初始化顺序

1. 安装 Docker 和 Docker Compose。
2. 拉取仓库并切换到 `cloud-center-mcp` 分支。
3. 在服务器本地创建 `warehouse.env`，填写密钥。
4. 启动 ClickHouse、PostgreSQL 和 MinIO。
5. 创建 bucket：`quant-assets`。
6. 初始化 ClickHouse database：`l1_raw`、`l2_base`、`l3_feature`、`l4_model`、`l5_strategy`、`l6_backtest`、`l7_trading`。
7. 初始化 PostgreSQL schema：`registry`、`audit`、`ops`。
8. 初始化 ClickHouse 用户、PostgreSQL 角色和最小权限。
9. 启动 MCP 子服务。
10. 执行健康检查。
11. 由审计智能体复核部署证据后，再允许生产资产迁移。

## 健康检查

ClickHouse：

```bash
docker compose ps clickhouse
curl "http://127.0.0.1:8123/?query=SELECT%201"
```

PostgreSQL：

```bash
docker compose ps postgres
docker exec -it <postgres_container> pg_isready -U quant_admin -d quant_warehouse
```

对象存储：

```bash
curl -I http://127.0.0.1:9100/minio/health/live
```

MCP 服务：

```bash
curl http://127.0.0.1:8101/health
```

registry 检查：

```sql
SELECT schema_name FROM information_schema.schemata
WHERE schema_name IN ('registry', 'audit', 'ops');
```

ClickHouse 分层检查：

```sql
SELECT name
FROM system.databases
WHERE name IN ('l1_raw', 'l2_base', 'l3_feature', 'l4_model', 'l5_strategy', 'l6_backtest', 'l7_trading');
```

## 本地到远程迁移原则

第一阶段只做镜像式发布，不改变本地主线默认读取入口。

迁移顺序：

1. 先迁移 PostgreSQL registry、manifest、审计报告和部署证据。
2. 再迁移 L1/L2/L3/L4/L5/L7 的 ClickHouse 只读业务表或对象存储快照。
3. 每个资产迁移后记录 hash、行数、分区范围和来源路径。
4. 审计智能体只读复核通过后，才能把远程资产标记为生产可读。
5. 主线默认读取入口切到远程前，必须单独审批。

## 风险与控制

| 风险 | 控制 |
| --- | --- |
| 未审计资产进入生产 | registry 必须绑定 audit record |
| 实验资产误入主线 | production / experimental 物理隔离，manifest 显式指定 |
| 多智能体越权写库 | 分 schema 分角色，MCP 子服务分权限 |
| ClickHouse 和 PostgreSQL 元数据不一致 | 生产资产写入必须先写业务表，再写 registry，并以 manifest 和审计记录做闭环校验 |
| 对象存储文件被覆盖 | 启用版本化，写入按版本目录 |
| 远程服务不可用 | 本地主线入口保留，远程先作为镜像和服务中台 |
| 密钥泄露 | `.env` 留在服务器本地，不入 Git，不进日志 |
| 数据不可回滚 | 每次迁移保留源路径、hash、版本和回滚记录 |

## 第一阶段交付边界

本阶段交付：

- 数仓智能体文档包。
- 远程数仓 MCP 中台方案文档。
- 可执行部署步骤和 schema 草案。
- Git 分支提交，供另一台服务器拉取。

本阶段不交付：

- 真实远程数据库创建。
- 真实生产资产迁移。
- 生产默认读取入口切换。
- 自动同步任务。
- 真实 MCP 服务代码实现。

以上事项需要后续单独审批和实施。
