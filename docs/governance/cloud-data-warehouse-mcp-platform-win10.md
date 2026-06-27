# Win10 远程数据中心落地方案

## 目标

本文档定义当前阶段基于 Win10 主机的远程数据中心落地方案。该方案服务于本项目的远程数仓 MCP 中台建设，重点解决以下问题：

- 在现有 Win10 主机上承载 ClickHouse、PostgreSQL、MinIO 和后续 MCP 子服务。
- 通过本机到服务器的 SSH 链路，让 Codex 能直接执行远程检查、部署和运维命令。
- 在不迁移生产资产、不切换主线默认入口的前提下，先建立远程中台和镜像式资产承载能力。
- 为后续迁移到标准 Linux 服务器保留平滑演进路径。

本文档是 `docs/governance/cloud-data-warehouse-mcp-platform.md` 的 Win10 落地补充，不替代总方案中的分层、权限和资产治理原则。

## 当前结论

当前环境建议采用两阶段策略：

1. 第一阶段：Win10 主机承担“远程中台试运行”和“部署验证”职责。
2. 第二阶段：如数据规模、并发或稳定性要求继续提升，再迁移到标准 Linux 服务器。

当前 Win10 方案的推荐定位是：

```text
Win10 + OpenSSH Server + Docker Desktop(Linux Containers) + ClickHouse + PostgreSQL + MinIO + MCP 子服务
```

这套方案适合先把远程数据中心搭起来，完成目录规范、端口规范、权限收口、资产 registry、审计闭环和 MCP 访问边界。它不是最终最优的长期生产宿主，但足够承担第一阶段远程中台角色。

## 约束与边界

- 当前只允许建设远程中台基础设施和治理面，不允许擅自迁移生产资产。
- 不允许切换本地主线默认读取入口到远端。
- 不允许未经审批启动生产自动同步、自动发布、生产预测、正式信号投递。
- 不输出真实密钥、密码、Token 或连接串。
- 涉及服务器实际部署、生产数据迁移、默认入口切换时，必须先经指挥官智能体归口并由主人审批。

## 主机角色定位

Win10 主机在当前阶段承担四类职责：

| 角色 | 说明 |
| --- | --- |
| 宿主机 | 承载 Docker Desktop 和基础容器 |
| 远程执行节点 | 通过 OpenSSH Server 接收本机发起的远程 PowerShell 命令 |
| 远程资产中台 | 承载 ClickHouse、PostgreSQL、MinIO 和后续 MCP 服务 |
| 验证环境 | 用于方案联调、目录规范固化、健康检查和部署演练 |

不建议 Win10 主机承担最终唯一的正式生产中心职责，尤其在出现以下条件时：

- ClickHouse 数据量持续增长并需要长期高吞吐写入。
- 多个 MCP 服务长期并发运行。
- 需要更稳定的备份、监控、计划任务和容器编排。
- 需要更严格的最小权限隔离和无桌面运维。

## 网络与控制面

### 远程控制链路

当前远程控制标准链路为：

```text
本机 Codex -> 本机 PowerShell -> SSH -> Win10 主机 -> PowerShell / Docker CLI
```

当前已验证可用的最小链路：

```powershell
ssh <win10-user>@<win10-host> "hostname"
ssh <win10-user>@<win10-host> "whoami"
ssh <win10-user>@<win10-host> "Get-ChildItem D:\"
```

这意味着后续所有只读检查、目录创建、文件同步和部署命令，都可以通过本机间接执行，无需来回手工导文件。

### 网络端口

当前建议端口口径如下：

| 服务 | 主机端口 | 说明 |
| --- | ---: | --- |
| SSH | 22 | 远程命令执行 |
| ClickHouse HTTP | 8123 | SQL HTTP 查询 |
| ClickHouse Native | 9000 | Native TCP |
| PostgreSQL | 5432 | 治理元数据库 |
| MinIO API | 9100 | 对象存储 API |
| MinIO Console | 9101 | 对象存储控制台 |
| asset-registry-mcp | 8101 | MCP skeleton 健康检查和元数据 |
| ops-mcp | 8102 | MCP skeleton 健康检查和元数据 |
| audit-mcp | 8103 | MCP skeleton 健康检查和元数据 |
| MCP Gateway 预留 | 8104-8109 | 后续子服务健康检查和 API |

原则：

- 当前阶段优先只开放局域网访问。
- 如未来存在跨网段访问，再通过 Tailscale / VPN 收口，不直接暴露到公网。
- MCP 服务端口不对外裸露给业务脚本，仍通过受控服务调用。

## 目录规划

当前 Win10 主机推荐使用 `D:` 盘作为数据中心根目录，避免系统盘承压。

建议目录：

```text
D:\quant\cloud-center\
  repo\
  env\
  volumes\
    clickhouse\
    postgres\
    minio\
    mcp_logs\
  backups\
  runtime\
  scripts\
```

职责说明：

- `repo\`：拉取当前 Git 仓库的 `cloud-center-mcp` 分支。
- `env\`：仅在服务器本地保存 `.env` 类敏感配置。
- `volumes\clickhouse\`：ClickHouse 数据目录。
- `volumes\postgres\`：PostgreSQL 数据目录。
- `volumes\minio\`：MinIO 对象数据目录。
- `backups\`：数据库导出、对象存储归档和恢复点。
- `runtime\`：运行期状态、健康检查输出、执行记录。
- `scripts\`：主机本地运维脚本和一次性操作脚本。

## 部署拓扑

### 基础组件

| 层 | 组件 | 当前定位 |
| --- | --- | --- |
| 存储计算 | ClickHouse | L1-L7 业务大表和分析查询 |
| 治理元数据 | PostgreSQL | registry / manifest / lineage / audit / ops |
| 对象存储 | MinIO | Parquet、模型、报告、归档、备份 |
| 服务入口 | MCP 子服务 | 统一受控访问层 |
| 远程控制 | OpenSSH Server | Codex 远程执行链路 |

### MCP 服务拆分规划

当前部署包已先提供 `asset-registry-mcp`、`ops-mcp`、`audit-mcp` 三个 skeleton 服务，用于验证容器拓扑、端口和健康检查。它们只提供 `/health`、`/metadata`、`/services`、`/tools`，不提供生产数据读写工具。

当前推荐至少拆分以下子服务：

| 服务 | 核心职责 | 默认写权限 |
| --- | --- | --- |
| `asset-registry-mcp` | registry、manifest、lineage、审计状态查询 | 数仓/审计受控写 |
| `l2-base-mcp` | L2 综合底表覆盖查询、范围查询、快照读取 | 数据整合写 |
| `l3-feature-mcp` | L3 因子、标签、manifest 查询 | 因子写 |
| `l4-model-mcp` | L4 预测结果、formal manifest、模型文件索引查询 | 模型写 |
| `audit-mcp` | 审计记录和证据索引查询 | 审计写 |
| `ops-mcp` | 任务状态、健康状态、部署记录、锁、日报查询 | 指挥官/数仓受控写 |

约束：

- MCP 按服务拆分，不做跨层万能写接口。
- 不允许任何单一服务同时拥有 L2-L7 多层写权限。
- 所有正式资产写入前后都必须能回链到 PostgreSQL `registry.assets` 与审计记录。

## Win10 宿主机实施方式

### 当前推荐实现

当前阶段建议：

1. Win10 主机安装 Docker Desktop。
2. 运行 Linux Containers 模式。
3. 通过 SSH 从本机远程执行 Docker、Git、PowerShell 命令。
4. 容器卷映射统一绑定到 `D:\quant\cloud-center\volumes\...`。

### 关键现实约束

当前仓库里的 `deploy/cloud-center/docker-compose.yml` 使用的是 Linux 宿主机路径：

```text
/opt/quant_mcp/volumes/clickhouse
/opt/quant_mcp/volumes/postgres
/opt/quant_mcp/volumes/minio
```

因此它当前适用于 Linux 服务器，不适合作为 Win10 主机的直接原样执行版本。Win10 上落地时需要单独准备 Windows 路径版 Compose 或通过环境替换统一收口路径。

这也是为什么当前阶段要把“总方案”和“Win10 落地方案”拆开写清。

## Win10 版部署建议

建议把 Win10 执行拆成三层：

### 第一层：主机准备

- 安装 Git。
- 安装 Docker Desktop，并确认可执行 `docker version`、`docker compose version`。
- 保持 OpenSSH Server 常驻运行。
- 为 `D:\quant\cloud-center\` 准备足够磁盘空间。

### 第二层：仓库和目录

```powershell
New-Item -ItemType Directory -Force D:\quant\cloud-center | Out-Null
cd D:\quant\cloud-center
git clone https://github.com/wangjrking/quant_vibe.git repo
cd repo
git checkout cloud-center-mcp
New-Item -ItemType Directory -Force D:\quant\cloud-center\env,D:\quant\cloud-center\volumes\clickhouse,D:\quant\cloud-center\volumes\postgres,D:\quant\cloud-center\volumes\minio,D:\quant\cloud-center\volumes\mcp_logs,D:\quant\cloud-center\backups,D:\quant\cloud-center\runtime | Out-Null
```

### 第三层：容器部署

Win10 版 Compose 文件需要把卷目录改成 Windows 路径，例如：

```text
D:/quant/cloud-center/volumes/clickhouse:/var/lib/clickhouse
D:/quant/cloud-center/volumes/postgres:/var/lib/postgresql/data
D:/quant/cloud-center/volumes/minio:/data
```

环境文件建议放在：

```text
D:\quant\cloud-center\env\warehouse.env
```

并保持真实密码和 Token 只存在主机本地，不进入 Git。

## 数据治理与资产边界

无论宿主机是 Win10 还是 Linux，数据治理口径不变：

- ClickHouse 放 L1-L7 业务明细和分析查询表。
- PostgreSQL 放 registry、manifest、lineage、audit、ops。
- MinIO 放 Parquet、模型、报告、快照、回测归档、交易归档和备份。
- 智能体默认不直连数据库，统一走 MCP。
- 生产资产与实验资产必须在存储路径、manifest 和权限上显式隔离。

推荐对象存储前缀保持：

```text
quant-assets/production/l1/
quant-assets/production/l2/
quant-assets/production/l3/
quant-assets/production/l4/
quant-assets/production/l5/
quant-assets/production/l6/
quant-assets/production/l7/

quant-assets/experimental/l1/
quant-assets/experimental/l2/
quant-assets/experimental/l3/
quant-assets/experimental/l4/
quant-assets/experimental/l5/
quant-assets/experimental/l6/
quant-assets/experimental/l7/
```

## 备份与恢复

Win10 阶段最少要准备以下备份闭环：

| 资产 | 备份方式 | 建议频率 |
| --- | --- | --- |
| PostgreSQL | `pg_dump` 导出到 `D:\quant\cloud-center\backups\postgres\` | 每日 |
| ClickHouse | 表级导出或快照归档到 `backups\clickhouse\` | 每日或每批次 |
| MinIO | 桶版本化 + 定期归档 | 每日 |
| env 与脚本 | 不入 Git，单独离线保存 | 变更后立即 |

恢复要求：

- 任一正式资产迁移前，必须先确认回滚路径。
- 备份文件必须附带时间戳、版本号、来源路径和责任人。
- 对象存储归档与 PostgreSQL registry 记录要能相互对应。

## 执行前检查清单

服务器执行前，至少检查以下内容：

1. `ssh <win10-user>@<win10-host> "hostname"` 能免密成功。
2. Win10 主机可执行 `docker version` 和 `docker compose version`。
3. `D:\quant\cloud-center\` 有足够剩余空间。
4. `22`、`8123`、`9000`、`5432`、`9100`、`9101` 端口未被冲突占用。
5. Docker Desktop 处于 Linux Containers 模式。
6. 环境变量文件只在主机本地落盘。
7. 备份目录已建立。
8. 尚未触发生产资产迁移、主线入口切换和自动同步。

## 分阶段实施建议

### 阶段 A：控制面打通

- 确认 SSH、Git、Docker、目录、端口、磁盘全部可用。
- 完成 Win10 路径版 Compose。
- 启动 ClickHouse、PostgreSQL、MinIO。
- 执行健康检查。

### 阶段 B：治理面落地

- 初始化 ClickHouse databases。
- 初始化 PostgreSQL `registry`、`audit`、`ops`。
- 固化对象存储 bucket 和 prefix 规范。
- 准备运维、审计和任务状态记录结构。

### 阶段 C：MCP 服务落地

- 先实现 `asset-registry-mcp`、`ops-mcp`、`audit-mcp`。
- 再实现 `l2-base-mcp`、`l3-feature-mcp`、`l4-model-mcp`。
- 所有服务先只做只读接口，再评估受控写接口。

### 阶段 D：资产镜像迁移

- 先迁治理元数据和部署证据。
- 再迁只读业务镜像表和对象存储快照。
- 审计复核通过后，再讨论生产可读标记。

## 风险说明

| 风险 | 当前控制 |
| --- | --- |
| Win10 宿主机稳定性弱于 Linux | 当前仅作为第一阶段远程中台和演练环境 |
| Docker Desktop 升级或重启影响服务 | 固定升级窗口，保留手工回滚和重启检查 |
| Compose 路径仍沿用 Linux 写法导致部署失败 | 必须单独维护 Win10 路径版 Compose |
| 误把远端验证环境当正式生产中心 | 在文档和审批流中明确阶段边界 |
| 真实密钥落入 Git | `env` 仅在服务器本地维护，不提交仓库 |

## 当前建议结论

当前最合理的路线不是推翻已有云端数仓方案，而是：

1. 保持总架构不变：`ClickHouse + PostgreSQL + MinIO + MCP`。
2. 把 Win10 主机作为第一阶段远程数据中心宿主。
3. 明确当前部署包是 Linux 口径，Win10 需要单独落地适配。
4. 先做远程控制面、基础存储面和治理面，不做生产资产切换。
5. 后续视规模和稳定性，再迁移到标准 Linux 服务器。
