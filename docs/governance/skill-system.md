# AGENT 入口与智能体内部技能治理说明

本文档定义本项目中 **AGENT 入口** 与 **智能体内部技能** 的关系，并明确不同类型规则应放在哪里。

## 核心结论

本项目不保留泛项目级业务 SKILL。

当前采用两层结构：

1. **AGENT 入口**：位于 `.codex/skills/<agent-id>/SKILL.md`，只用于 Codex 平台识别并路由到某个智能体。
2. **智能体内部技能**：位于 `.codex/agent_packages/<agent-id>/skills.md`，定义该智能体自己的可复用能力入口。

目录名 `.codex/skills/` 保留，是因为这是 Codex 平台识别入口的实现约束；但语义上不能把 AGENT 入口称为业务技能。

## AGENT 入口

位置：

```text
.codex/skills/<agent-id>/SKILL.md
```

职责：

- 描述什么场景触发该智能体。
- 指向对应的 `.codex/agent_packages/<agent-id>/`。
- 保留必要边界，避免误触发业务链路。

不得写入：

- 完整工作流。
- 长篇业务规则。
- 历史事故补丁。
- 专项表字段口径。
- 生产资产细节。
- 智能体内部日志、工具和知识库全文。

当前 AGENT 入口包括：

```text
commander-agent
architect-agent
audit-agent
data-ingestion-agent
data-integration-agent
factor-agent
model-agent
strategy-agent
trading-agent
research-agent
```

## 智能体内部技能

位置：

```text
.codex/agent_packages/<agent-id>/skills.md
```

职责：

- 记录具体、可重复调用的智能体能力入口。
- 说明能力名称和归属，不展开长流程。
- 与 `capabilities.md`、`procedures.md`、`permissions.md` 保持一致。

当前从平台层迁入 `factor-agent` 的内部技能包括：

- 因子治理审计。
- 因子增量更新。
- GTJA 公式审计。

## 放置规则

| 内容类型 | 归档位置 | 说明 |
| --- | --- | --- |
| 智能体识别与路由 | `.codex/skills/<agent-id>/SKILL.md` | 只做平台入口，不承载业务细节 |
| 可重复能力入口 | `.codex/agent_packages/<agent-id>/skills.md` | 只写“能做什么”，不写长流程 |
| 能力边界与职责范围 | `.codex/agent_packages/<agent-id>/capabilities.md` | 写清智能体能负责哪些层和资产 |
| 输入输出契约 | `.codex/agent_packages/<agent-id>/contract.md` | 标准任务格式、回执格式、证据格式、失败码 |
| 执行步骤 | `.codex/agent_packages/<agent-id>/procedures.md` | 写“如何做”，包括前置条件和作业规程 |
| 权限与停止条件 | `.codex/agent_packages/<agent-id>/permissions.md` | 写允许、禁止、需审批和升级条件 |
| 工具与脚本 | `.codex/agent_packages/<agent-id>/tools.md` | 写可用工具、脚本入口和禁止使用的脚本 |
| 知识与资产索引 | `.codex/agent_packages/<agent-id>/knowledge.md` | 写资产路径、报告、字段口径、历史证据 |
| 交接与回执 | `.codex/agent_packages/<agent-id>/handoff.md` | 写跨智能体交接、送审、回传和闭环规则 |
| 日志与监控 | `.codex/agent_packages/<agent-id>/logging.md` | 写日志位置、状态字段、耗时、卡点和监控要求 |
| 执行历史 | `quant/data_file/runtime/agent_memory/agent_notes/` | 只放运行留痕和阶段性记录 |
| 工作流实例 | `quant/data_file/runtime/orchestrator_reports/workflow_monitor/` | 指挥官创建和维护任务状态 |
| 完整跨层工作流 | `WORKFLOW.md` 与 `.codex/agent_packages/commander-agent/workflows.md` | 只有指挥官调度完整工作流 |

## 智能体文档包

位置：

```text
.codex/agent_packages/<agent-id>/
```

职责：

- `README.md`：智能体介绍和职责定位。
- `capabilities.md`：能力清单。
- `skills.md`：智能体内部技能入口。
- `contract.md`：输入输出契约。
- `permissions.md`：权限边界。
- `procedures.md`：标准作业规程。
- `tools.md`：可用工具和脚本。
- `knowledge.md`：知识库索引。
- `logging.md`：日志和留痕规则。
- `handoff.md`：交接规则。
- `audit-checklist.md`：自检和送审清单。
- `codex-automations.md`：Codex 自动化任务定义。

## 读取顺序

智能体被触发后，推荐读取顺序为：

1. `.codex/skills/<agent-id>/SKILL.md`
2. `.codex/agent_packages/<agent-id>/README.md`
3. `.codex/agent_packages/<agent-id>/capabilities.md`
4. `.codex/agent_packages/<agent-id>/skills.md`
5. `.codex/agent_packages/<agent-id>/procedures.md`
6. `.codex/agent_packages/<agent-id>/permissions.md`

如任务涉及工具、资产、交接或日志，再读取 `tools.md`、`knowledge.md`、`handoff.md`、`logging.md`。

## 修改规则

- 修改触发场景：改 AGENT 入口 `SKILL.md` 的 frontmatter description。
- 修改智能体职责、流程、权限、工具、知识库：改 `.codex/agent_packages/<agent-id>/`。
- 修改智能体内部技能：改 `.codex/agent_packages/<agent-id>/skills.md`。
- 修改跨智能体共用规则：改 `.codex/agent_packages/common-rules.md`。
- 修改完整工作流：改 `WORKFLOW.md` 或 `.codex/agent_packages/commander-agent/workflows.md`。
- 修改线程发送和审计回执规则：改 `.codex/agents/communication-layer.md`。

## 防散乱规则

- 一个规则只能有一个主归档位置，其他文件只做索引引用。
- 新规则超过 30 行时，不得写入 `quant/main/AGENTS.md` 或 AGENT 入口 `SKILL.md`。
- 智能体内部技能必须归属到具体智能体文档包，不得放成项目泛化说明。
- AGENT 入口只允许保留路由信息和最小边界。
- 若执行中发现重复任务、卡点或技能缺陷，应优先完善对应智能体的 `skills.md`、`procedures.md` 或辅助校验脚本，而不是把规则堆回平台入口。
