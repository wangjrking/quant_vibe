# 项目目录导航

本文档给出 Quant MCP 项目的推荐阅读顺序和目录定位。它是面向人和智能体的结构导航，不承载具体业务规则。

## 推荐阅读顺序

1. `WORKFLOW.md`
2. `quant/main/AGENTS.md`
3. `quant/main/docs/governance/project-doc-map.md`
4. `quant/main/docs/governance/project-structure-reorganization.md`
5. `.codex/agents/README.md`
6. `.codex/agents/communication-layer.md`
7. 与任务相关的 `.codex/agent_packages/<agent-id>/README.md`

## 目录定位

| 路径 | 定位 | 默认处理原则 |
| --- | --- | --- |
| `.codex/agents/` | 智能体角色卡、通信层和专项协同规则 | 简洁入口，不承载长规则 |
| `.codex/agent_packages/` | 智能体完整职责、权限、工具、流程和审计清单 | 每个智能体的正式制度包 |
| `.codex/skills/` | 平台 AGENT 路由入口 | 只放正式智能体触发、路由和简要说明，不承载业务技能细节 |
| `quant/main/` | 当前代码主目录 | 根目录逐步瘦身，主线入口优先 |
| `quant/main/core/` | 可复用核心代码 | 不放一次性研究脚本 |
| `quant/main/workflows/` | 标准工作流入口 | 只放经过审查的主线编排 |
| `quant/main/tools/` | 治理、检查、辅助工具 | 不默认触发业务链路 |
| `quant/main/research/` | 研究、调参、候选探索脚本 | research-only，不作为生产入口 |
| `quant/main/legacy/` | 旧链路、回滚、历史复现入口 | 必须显式 legacy，默认不使用 |
| `quant/main/docs/governance/` | 项目治理文档 | 新增或改写 Markdown 默认中文 |
| `quant/data_file/reports/` | 审计、模型、策略、数据报告 | 作为证据保留，不作为规则主入口 |
| `quant/data_file/runtime/` | 运行态记忆、监控、回收站和临时留痕 | 按 runtime 治理规则处理 |

## 主线资产入口

| 层级 | 当前默认入口 |
| --- | --- |
| L1 原始数据 | `quant/data_file/raw_table_dbs/[table].DB` |
| L2 综合底表 | `quant/data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA` |
| L3 生产特征 | `quant/data_file/production_factor_parts/` |
| L3 标签 | `quant/data_file/prediction_label_parts/` |
| L4 正式预测 | `quant/data_file/model_predictions/` 与 approved formal manifest |
| L5/L6 策略与回测 | `quant/main/strategy_library/` 和标准 L5 入口 |
| L7 交易交付 | 交易智能体维护的平台交付目录 |

## 不应作为默认入口的资产

以下资产只能用于 legacy、审计、回滚或历史复现，不应作为新链路默认输入：

- `quant/data_file/odb.db`
- `quant/data_file/stock_factor_data.parquet`
- `odb.db.stock_predict_data_*`
- 未审批的 research-only manifest
- 临时生成的回填 parquet、临时 score 表或调参中间表

## 查找建议

查业务链路，先看 `WORKFLOW.md` 和 `quant/main/AGENTS.md`。

查智能体职责，先看 `.codex/agents/README.md`，再看 `.codex/agent_packages/<agent-id>/`。

查目录和文档归口，先看 `quant/main/docs/governance/project-doc-map.md` 和本文档。

查某个脚本是否应保留，先看 `quant/main/docs/governance/project-structure-reorganization.md` 的 Keep / Move / Archive / Move to recycle bin 规则。
