# 项目结构整理方案

本文档定义 Quant MCP 项目的目录整理目标、迁移规则和执行边界。它用于指导指挥官智能体、架构师智能体、审计智能体和各专业智能体后续进行文件治理。

## 目标

本轮整理采用分层重构方案，不追求一次性清空所有历史痕迹。目标是让项目形成稳定入口、清晰分区和可追溯的运行留痕。

整理后的项目应满足：

- 新成员可以先从固定入口理解项目，不需要在大量脚本中寻找主线。
- 当前标准链路、研究脚本、legacy 复现入口和临时运行产物有明确边界。
- 文件迁移、归档和回收都能保留原路径、动作原因和责任智能体。
- 不破坏审计报告、策略归档、manifest 和线程台账中已有的证据链。

## 总体分层

| 层级 | 目录或文件 | 定位 |
| --- | --- | --- |
| 主入口层 | `WORKFLOW.md`、`quant/main/AGENTS.md`、`quant/main/docs/governance/project-doc-map.md` | 项目阅读和调度入口 |
| 智能体治理层 | `.codex/agents/`、`.codex/agent_packages/`、`.codex/skills/` | 智能体职责、线程、权限和技能入口 |
| 标准业务代码层 | `quant/main/core/`、`quant/main/workflows/`、`quant/main/tools/`、`quant/main/config/`、`quant/main/strategy_library/` | 当前标准链路、工具和正式资产入口 |
| 研究与历史层 | `quant/main/research/`、`quant/main/legacy/`、必要的 archive 子目录 | 研究、调参、历史复现和旧链路 |
| 文档治理层 | `quant/main/docs/governance/` | 结构、文档、线程、资产、runtime 等治理规则 |
| 运行态留痕层 | `quant/data_file/runtime/` | 智能体记忆、报告、回收站、运行状态和临时复现证据 |

## 文件处置口径

| 判定 | 适用对象 | 处理方式 |
| --- | --- | --- |
| Keep | 当前标准链路、测试、manifest、审计报告、策略归档、线程治理仍引用的文件 | 保留原路径，必要时补中文说明 |
| Move | 文件仍有价值，但当前目录位置错误 | 移到 `research/`、`legacy/`、`tools/`、`workflows/` 等更合适位置 |
| Archive | 需要长期留痕，但不应作为主干入口暴露 | 移入 archive 或 backups 目录，并补中文 `README.md` |
| Move to recycle bin | 无当前链路、测试、文档、审计、归档依赖，且可等待观察期 | 使用项目回收站，观察 30 天后再清理 |

## `quant/main` 根目录治理

`quant/main` 根目录只应保留以下类型：

- 当前标准入口脚本。
- 当前标准模块和被大量导入的公共模块。
- 项目配置、README、requirements 等基础文件。
- 尚未完成归口审计的过渡文件。

以下类型应逐步迁出根目录：

- `research_*`、`tune_*`、`optimize_*`、`probe_*`、`check_*`、`analyze_*` 等研究和调参脚本。
- 带明确日期的一次性验证、发布、归档、候选策略脚本。
- 历史修补、旧链路回滚、兼容复现脚本。
- 临时测试文件、缓存目录和可再生成产物。

迁出前必须先检查：

- 是否被当前 Python 模块、测试、配置、manifest 或调度脚本导入。
- 是否被 `WORKFLOW.md`、`AGENTS.md`、README、审计报告或策略归档显式引用。
- 是否承担 legacy rollback 或 historical reproduction 职责。

## `runtime` 治理

`runtime` 是运行留痕区，不按源码目录方式整理。核心原则是保留可追溯性，同时降低临时目录对主视野的干扰。

长期保留目录：

- `agent_memory/`
- `agent_observability/`
- `agent_workspaces/`
- `orchestrator_reports/`
- `recycle_bin/`
- `trading_agent/`

阶段归档目录：

- 复现实验、snapshot、strategy replay、debug 目录如仍有证据价值，应移入 `runtime/archive/` 并保留说明。

回收站候选：

- 明确无证据价值、可再生成、无引用的 `tmp_*`、缓存、测试产物和临时日志目录。

## 执行顺序

1. 固定文档入口和目录说明。
2. 生成 `quant/main` 根目录和 `runtime` 的整理台账。
3. 先处理高置信度缓存和临时文件。
4. 对脚本迁移候选分配给责任智能体认领。
5. 审计智能体复核误删、误迁和证据链风险。
6. 批量迁移前再次汇总给用户确认。

## 权限边界

整理动作不得触发业务脚本，不得补数据、算因子、训练模型、生成预测、生成信号或跑回测。

不得修改投产策略参数、正式 manifest、生产任务配置或审计结论。

涉及生产入口、legacy 必保留入口、策略归档引用文件和审计证据文件的迁移，必须先形成台账并经审计复核。
