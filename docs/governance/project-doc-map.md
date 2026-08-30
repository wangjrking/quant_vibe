# 项目文档地图

本文档说明本项目 Markdown 文档的分层、主归档位置和读取顺序。

## 总原则

- 一个规则只保留一个主归档位置，其他文件只放索引或引用。
- AGENT 入口 `SKILL.md` 只负责平台发现和路由；具体可复用能力由智能体内部技能承载。
- 智能体完整职责、技能、权限、工具、流程、知识库、日志、交接和监控规则统一放入 `.codex/agent_packages/<agent-id>/`。
- 运行证据和审计报告保留在 `quant/data_file/reports/` 或对应运行目录，不直接混入治理文档。

## 必读入口

### 项目级

1. `WORKFLOW.md`
2. `quant/main/AGENTS.md`
3. `.codex/agents/README.md`
4. `.codex/agents/communication-layer.md`
5. `.codex/agent_packages/common-rules.md`

### 智能体级

1. `.codex/skills/<agent-id>/SKILL.md`
2. `.codex/agent_packages/<agent-id>/README.md`
3. `.codex/agent_packages/<agent-id>/capabilities.md`
4. `.codex/agent_packages/<agent-id>/skills.md`
5. `.codex/agent_packages/<agent-id>/procedures.md`
6. `.codex/agent_packages/<agent-id>/permissions.md`

## 文档分区

| 目录或文件 | 定位 | 维护方式 |
| --- | --- | --- |
| `WORKFLOW.md` | 主线工作流定义 | 项目级治理文档 |
| `quant/main/AGENTS.md` | 轻量入口和硬规则索引 | 只放入口级规则 |
| `.codex/agents/` | 角色卡、通信协议、专项协作规范 | 智能体启动层 |
| `.codex/skills/` | 平台 AGENT 入口 | 只放智能体路由和触发描述 |
| `.codex/agent_packages/` | 智能体完整文档包 | 职责、技能、权限、流程、工具、日志、交接 |
| `quant/main/docs/governance/` | 项目治理文档 | 架构、资产、技能、文档、GitHub、云中台 |
| `quant/main/docs/governance/project-structure-guide.md` | 项目目录导航 | 说明阅读顺序和主要目录定位 |
| `quant/main/docs/governance/project-structure-reorganization.md` | 项目结构整理方案 | 说明 Keep / Move / Archive / Move to recycle bin 规则 |
| `quant/main/docs/governance/runtime-governance.md` | runtime 目录治理规则 | 说明运行态目录保留、归档和回收规则 |
| `quant/main/docs/governance/duckdb-materialization-readiness.md` | DuckDB 真实物化前准备规范 | 定义真实物化前送审包、证据字段、责任归口和审批边界 |
| `quant/main/docs/governance/mcp-asset-gateway-platform.md` | MCP 资产网关中台方案 | 当前 L8 MCP 资产发布层主线，说明 PostgreSQL、MinIO/对象存储、MCP gateway、发布资产、权限和审计边界 |
| `quant/main/docs/governance/cloud-data-warehouse-mcp-platform.md` | 历史远程数仓 MCP 中台方案 | 旧 ClickHouse 全量迁移和远程数据中心方向，当前仅作历史参考 |
| `quant/main/docs/governance/history/` | 历史治理快照 | 只保留迁移前规则快照 |
| `quant/data_file/reports/` | 审计、模型、策略、数据报告 | 运行证据，不作为规则主入口 |
| `quant/data_file/runtime/` | 运行状态、监控、记忆、回收站 | 运行态记录 |
| `quant/main/strategy_library/` | 策略生产和探索归档 | 策略证据与复现材料 |

## 修改归口

| 修改内容 | 主归档位置 |
| --- | --- |
| 完整工作流 | `WORKFLOW.md`、`.codex/agent_packages/commander-agent/workflows.md` |
| 智能体职责和权限 | `.codex/agent_packages/<agent-id>/README.md`、`permissions.md` |
| 智能体内部技能 | `.codex/agent_packages/<agent-id>/skills.md` |
| 平台触发描述 | `.codex/skills/<agent-id>/SKILL.md` |
| 跨线程通信和回执模板 | `.codex/agents/communication-layer.md` |
| 线程制智能体管理、线程注册表和自动化绑定规则 | `quant/main/docs/governance/thread-based-agent-management.md` |
| 项目目录导航、结构整理、根目录瘦身和 runtime 留痕治理 | `quant/main/docs/governance/project-structure-guide.md`、`quant/main/docs/governance/project-structure-reorganization.md`、`quant/main/docs/governance/runtime-governance.md` |
| 正式资产和实验资产边界 | `quant/main/docs/governance/standard-agent-architecture.md`、资产 registry |
| Markdown 文档治理 | `quant/main/docs/governance/markdown-document-registry.md` |
| 旧规则快照 | `quant/main/docs/governance/history/` |

## 冗余处理规则

- 过期治理文档不得直接删除，先迁移到项目回收站。
- 运行证据报告不得按“数量多”判断为冗余。
- 重复模板和空占位 README 应合并到上级 README，子目录只保留目录本身。
- 回收站观察期为 30 天。
