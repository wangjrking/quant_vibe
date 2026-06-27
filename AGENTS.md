# AGENTS.md

本文件是 `quant/main/` 目录下 Codex 智能体与脚本治理的轻量入口。它不再承载完整业务规则、历史补丁或专项链路说明。

完整历史内容已保留在：

`quant/main/docs/governance/history/AGENTS_legacy_20260625.md`

## 定位

- 本文件只保存最高优先级入口规则、阅读顺序、硬边界和文档分流。
- 智能体完整职责、技能、权限、工具、知识库、日志、交接和监控规则，统一存放在 `.codex/agent_packages/<agent-id>/`。
- 平台 AGENT 入口 `.codex/skills/<agent-id>/SKILL.md` 只负责平台识别和路由，不承载完整智能体说明；可复用业务能力必须写入对应智能体文档包的内部技能。
- 专项治理规则必须进入对应专题文档，不得继续堆入本文件。

## 必读顺序

执行 `quant/main/` 相关任务前，按需阅读：

1. `.codex/agents/README.md`
2. `.codex/agents/communication-layer.md`
3. `.codex/agent_packages/common-rules.md`
4. `.codex/agent_packages/<agent-id>/README.md`
5. `.codex/agent_packages/<agent-id>/capabilities.md`
6. `.codex/agent_packages/<agent-id>/procedures.md`
7. `WORKFLOW.md`
8. 与任务相关的专题治理文档

## 最高优先级硬规则

- 所有项目内新增或改写的 Markdown 文档默认使用中文；代码、命令、路径、字段名和英文专有名词可保留原文。
- 文档说明和资产清单默认使用项目相对路径；命令示例、统一解释器和真实证据路径可使用本机绝对路径。详细规则见 `.codex/agent_packages/common-rules.md`。
- 环境路径必须明确写成本机绝对路径；所有运行统一使用 `D:/work/quant/quant_mcp/.venv/Scripts/python.exe`。详细环境规则见 `.codex/agent_packages/common-rules.md`。
- 不得提交、打印或泄露真实 Token、账号、密钥、私钥或未脱敏登录信息。
- 正式资产发生任何变动前，必须先经过审计智能体审计并留存记录。
- 写入 `collaboration_requests.jsonl` 或 `agent_notes/` 只代表通信层留痕，不等于已经发送到目标智能体线程。
- 其他智能体向审计智能体发起送审后，审计完成必须回传原发起线程；只有回传完成才算跨线程审计闭环完成。
- 处理“今天、昨天、最新、下一交易日”等需求时，必须先明确自然日、目标交易日、最新已完成交易日、信号日和执行日。
- 不得把 legacy 资产、实验资产或临时研究资产作为主线默认输入，除非有显式 manifest、审计记录和用户/指挥官批准。

## 分层边界

当前主线链路采用 L0-L8：

| 层级 | 中文名称 | 主要责任智能体 |
| --- | --- | --- |
| L0 | 配置、环境与运行状态层 | 指挥官、架构师、审计 |
| L1 | 原始数据层 | 数据接入智能体 |
| L2 | 综合底表层 | 数据整合智能体 |
| L3 | 因子与标签层 | 因子智能体 |
| L4 | 模型调优与预测资产层 | 模型智能体 |
| L5 | 策略信号层 | 策略智能体 |
| L6 | 回测验证层 | 策略智能体 |
| L7 | 交易交付层 | 交易智能体 |
| L8 | 多智能体协同治理层 | 指挥官、架构师、审计 |

数仓智能体属于 L0/L8 基础设施与治理支撑角色，负责远程数据库、对象存储、云端数仓、MCP 数据服务、权限、备份、同步和运维方案，不直接生产 L1-L7 业务资产。

跨层任务必须由指挥官智能体分派；专业智能体不得越权推进下游正式链路。

## 标准资产入口

- L1 raw split DB：`quant/data_file/raw_table_dbs/[table].DB`
- L2 综合日线底表：`quant/data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA`
- L3 生产特征：`quant/data_file/production_factor_parts/`
- L3 训练标签：`quant/data_file/prediction_label_parts/`
- L4 正式预测资产：必须通过 `approved_for_l5` formal manifest 声明
- L5/L6 策略与回测资产：必须进入策略库或正式归档
- L7 交易交付资产：必须由交易智能体按平台交付口径管理

`odb.db`、`stock_factor_data.parquet`、`odb.db.stock_predict_data_*` 和旧 mixed-db 入口均为 legacy/审计/回滚/复现场景资产，不是新链路默认入口。

## 文档分流规则

新增或修改规则时按以下位置归档：

| 内容类型 | 归档位置 |
| --- | --- |
| 跨智能体协作、线程发送、回执闭环 | `.codex/agents/communication-layer.md` |
| 线程制智能体管理、线程注册表字段、线程生命周期 | `quant/main/docs/governance/thread-based-agent-management.md` |
| 项目目录导航、结构整理、根目录瘦身、runtime 治理 | `quant/main/docs/governance/project-structure-guide.md`、`quant/main/docs/governance/project-structure-reorganization.md`、`quant/main/docs/governance/runtime-governance.md` |
| 全体智能体通用硬规则 | `.codex/agent_packages/common-rules.md` |
| 智能体完整职责、技能、权限、工具、日志、监控 | `.codex/agent_packages/<agent-id>/` |
| 工作流定义和指挥官调度 | `WORKFLOW.md`、`.codex/agent_packages/commander-agent/workflows.md` |
| 资产分层、生产/实验资产、manifest、审计门禁 | `quant/main/docs/governance/` |
| 远程数据库、对象存储、云端数仓和 MCP 数据服务 | `quant/main/docs/governance/cloud-data-warehouse-mcp-platform.md` |
| 策略生产准入、风格暴露、低路径依赖 | `.codex/agents/strategy-admission-standards.md` 或策略治理专题文档 |
| 旧规则、历史补丁、事故经验 | `quant/main/docs/governance/history/` |
| 工具和校验脚本说明 | `quant/main/tools/README.md` |

## 修改本文件的规则

- 只有入口级、全局级、不可再分流的规则才允许写入本文件。
- 新增超过 30 行的规则，必须优先创建或更新专题文档，然后在本文件只放索引。
- 修改本文件后，至少运行：

```powershell
D:/work/quant/quant_mcp/.venv/Scripts/python.exe D:/work/quant/quant_mcp/quant/main/tools/agent_governance_check.py
D:/work/quant/quant_mcp/.venv/Scripts/python.exe D:/work/quant/quant_mcp/quant/main/tools/standard_agent_architecture_check.py
```
