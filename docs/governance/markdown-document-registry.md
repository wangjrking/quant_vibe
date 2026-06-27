# Markdown 文档注册表

本文档定义项目内 Markdown 文件的归类、保留规则和清理边界。

## 文档类型

| 类型 | 位置 | 是否可直接删除 | 说明 |
| --- | --- | --- | --- |
| 项目工作流 | `WORKFLOW.md` | 否 | 主线工作流入口 |
| 工程入口规则 | `quant/main/AGENTS.md` | 否 | 轻量入口和硬规则索引 |
| 智能体角色卡 | `.codex/agents/` | 否 | 启动角色、通信协议和专项协作规范 |
| 平台 AGENT 入口 | `.codex/skills/<agent-id>/SKILL.md` | 否 | 平台识别和路由；不是业务技能 |
| 智能体文档包 | `.codex/agent_packages/<agent-id>/` | 否 | 完整职责、技能、权限、流程、工具、日志、交接 |
| 治理文档 | `quant/main/docs/governance/` | 否 | 架构、资产、技能、文档、云中台等治理方案 |
| 历史治理快照 | `quant/main/docs/governance/history/` | 可归档 | 保留迁移前规则语义 |
| 审计和运行报告 | `quant/data_file/reports/` | 否 | 证据链，不能只因数量多删除 |
| 运行态记录 | `quant/data_file/runtime/` | 视规则处理 | 监控、记忆、回收站、交易交付记录 |
| 策略归档 | `quant/main/strategy_library/` | 否 | 投产和探索策略证据 |
| 实验资产说明 | `quant/data_file/experimental_assets/` | 视资产状态处理 | 实验轨道说明 |
| 生产资产说明 | `quant/data_file/production_assets/` | 否 | 生产轨道说明 |

## 冗余识别

优先识别以下冗余：

- 乱码或不可读治理文档。
- 与上级 README 完全重复的占位 README。
- 平台级技能旧资源目录中的过期 `checklists/`、`examples/`、`references/`。
- 只写“详见 capabilities.md”的空壳 `skills.md`。
- 已迁移到专题文档后仍残留的历史规则块。

不直接视为冗余：

- 审计报告。
- 生产策略归档。
- 模型、策略、交易运行证据。
- 回收站内观察期文件。

## 清理流程

1. 先确认文件是否仍被当前链路、文档、测试或审计证据引用。
2. 可合并的先合并到上级 README 或专题文档。
3. 确认无用后，使用 `quant/main/tools/recycle_bin.py move` 移入项目回收站。
4. 回收站记录必须包含 actor、reason、original_path、trashed_path 和时间。
5. 观察期 30 天，未恢复、未再次修改后才允许自动清理。

## 当前规范

- 所有新增或改写 Markdown 默认使用中文。
- 代码、命令、路径、字段名、表名、模型名和英文专有名词可保留原文。
- 新增规则超过 30 行时，不得写入 `AGENTS.md` 或 AGENT 入口 `SKILL.md`，应进入专题治理文档或智能体文档包。
