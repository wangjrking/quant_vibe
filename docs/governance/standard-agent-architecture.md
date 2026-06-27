# 标准智能体架构建设方案

## 定位

本文档定义本项目从“多智能体协作项目”升级为“标准多智能体平台项目”的目标架构、当前不足、分阶段补齐路线和落地校验方式。

本方案不替代 `WORKFLOW.md`、`quant/main/AGENTS.md` 和 `.codex/agents/communication-layer.md`。当规则冲突时，仍按项目文档优先级处理：

1. 主人明确指令
2. `WORKFLOW.md`
3. `quant/main/AGENTS.md`
4. `.codex/agents/communication-layer.md`
5. `.codex/agent_packages/<agent-id>/`
6. `.codex/skills/<agent-id>/`

## 当前主要不足

### 规则仍偏文档化

当前项目已有较完整的规则文档，但生产资产变更、审计前置、线程闭环、工作流状态等要求仍主要依赖智能体自觉执行。标准架构应把关键规则升级为可运行校验。

### 工作流状态机还不完整

指挥官已有工作流监控模板，但每次任务尚未强制形成统一任务实例。需要用固定字段记录任务状态、负责人、开始时间、预计完成时间、阻塞原因、审计状态和线程闭环状态。

### 资产注册中心尚未成为唯一入口

项目已经有 `production_assets.json`、`experimental_assets.json` 和 `manifest.schema.json`，但目前仍处于骨架阶段。标准架构要求正式资产和实验资产都必须先注册，再进入下游使用。

### 智能体输入输出契约不足

每个智能体已有 `capabilities.md`、`permissions.md`、`procedures.md`、`handoff.md` 等文档，但还缺少统一 `contract.md`，用于定义可接收任务类型、必需输入、标准输出、失败码和交接条件。

### 运行态监控未产品化

当前有 `agent_notes`、`collaboration_requests.jsonl`、`decisions.json` 和 workflow monitor 模板，但缺少统一汇总、超时检查、卡点统计和审计闭环状态视图。

### 主仓研究脚本噪音仍高

`quant/main` 仍存在大量研究、调参、候选验证和历史脚本。标准架构要求将主线工作流入口、研究探索脚本、legacy 入口和治理工具物理隔离，降低误用概率。

## 目标五层架构

### 交互层

对象：

- 主人
- 指挥官智能体
- 各专业智能体线程

职责：

- 接收需求
- 匹配工作流
- 展示任务进度
- 返回可核对结果

### 治理层

对象：

- `WORKFLOW.md`
- `quant/main/AGENTS.md`
- `.codex/agents/communication-layer.md`
- 审计前置规则
- 交易日规则
- 权限与审批规则

职责：

- 定义主线工作流
- 定义 L0-L8 分层边界
- 定义审计、审批、回执和越权拦截规则

### 执行层

对象：

- `.codex/skills/<agent-id>/SKILL.md`
- `.codex/agent_packages/<agent-id>/capabilities.md`
- `.codex/agent_packages/<agent-id>/procedures.md`
- `quant/main/tools/`
- 校验脚本

职责：

- 平台识别智能体
- 智能体按项目规则执行
- 用工具校验规则是否落地

### 资产层

对象：

- `quant/data_file/production_assets/`
- `quant/data_file/experimental_assets/`
- `quant/data_file/asset_registry/`
- manifests
- lineage
- audit records

职责：

- 区分生产资产和实验资产
- 记录资产来源、状态、审计、归属和下游许可
- 禁止未注册资产进入主线默认输入

### 监控层

对象：

- `quant/data_file/runtime/orchestrator_reports/workflow_monitor/`
- `quant/data_file/runtime/agent_memory/`
- 指挥官日报
- 卡点和超时记录

职责：

- 记录任务状态
- 记录审计闭环
- 记录资产变更
- 记录阻塞、超时和待审批事项

## 分阶段补齐路线

### 第一阶段：把规则从文档升级成校验

目标：

- 资产 registry 真正启用
- 审计前置要求可检查
- 生产资产变更可拦截
- 线程闭环状态可检查
- workflow monitor 自动状态字段可校验

本阶段落地项：

- 新增 `quant/main/tools/standard_agent_architecture_check.py`
- 校验资产 registry 基础结构
- 校验生产资产必须有审计记录
- 校验实验资产不得允许主线默认读取
- 校验 workflow monitor 模板必须包含审计闭环字段
- 校验每个智能体必须有 `capabilities.md`

### 第二阶段：统一智能体 I/O 契约

目标：

- 每个智能体补齐 `contract.md`
- 指挥官按契约分派任务
- 专业智能体按契约回执

每个 `contract.md` 至少包含：

- `accepted_task_types`
- `required_inputs`
- `optional_inputs`
- `forbidden_actions`
- `produced_artifacts`
- `handoff_conditions`
- `audit_required_when`
- `failure_codes`

### 第三阶段：运行态半产品化

目标：

- 工作流任务实例化
- 任务状态可汇总
- 卡点和超时可发现
- 审计闭环可追踪
- 生产资产变更可回看

建议落地项：

- 指挥官维护 `workflow_state.jsonl`
- 每次任务生成唯一 `task_id`
- 每个专业智能体回报时更新状态字段
- 每日自动生成治理日报

### 第四阶段：主仓噪音治理

目标：

- 降低误用历史脚本、研究脚本和 legacy 入口的概率

建议目录：

```text
quant/main/core/
quant/main/workflows/
quant/main/agent_tools/
quant/main/research/
quant/main/legacy/
```

处理原则：

- 当前主线入口进入 `workflows/`
- 可复用核心模块进入 `core/`
- 智能体治理工具进入 `agent_tools/` 或保留 `tools/`
- 研究脚本进入 `research/`
- 旧链路入口进入 `legacy/`
- 移动前必须做引用扫描和审计复核，不能直接物理删除

## 最小合格标准

一个标准智能体架构任务至少应满足：

- 指挥官明确匹配工作流
- 明确 owner agent
- 明确目标层级和交易日口径
- 明确是否涉及生产资产变更
- 涉及生产资产变更时必须先审计
- 涉及实验资产时必须显式 manifest
- 涉及跨线程审计时必须有回执闭环状态
- 完成后有证据路径和验证结果

## 当前第一阶段状态

已建立：

- 标准架构总文档：`quant/main/docs/governance/standard-agent-architecture.md`
- 第一阶段只读校验脚本：`quant/main/tools/standard_agent_architecture_check.py`

## 当前第二阶段状态

已建立：

- 每个智能体文档包均已补齐 `contract.md`
- 治理检查脚本已要求每个智能体必须具备 `contract.md`
- `contract.md` 作为指挥官任务分派和专业智能体回执的输入输出契约

## 当前第三阶段状态

已建立：

- 工作流监控汇总脚本：`quant/main/tools/workflow_monitor_report.py`
- 工作流监控管理脚本：`quant/main/tools/workflow_monitor_manage.py`
- 该脚本只读扫描 `quant/data_file/runtime/orchestrator_reports/workflow_monitor/*.json`
- 输出工作流状态、智能体任务状态、阻塞数量、生产资产变更状态和审计线程闭环状态
- 指挥官启动多智能体协作、审计、审批、生产资产变更或超过单步问答的任务时，应先创建 workflow monitor 实例。
- 专业智能体阶段性回报、阻塞、审计回执和完工结果，应由指挥官更新到 workflow monitor。

## 当前第四阶段状态

已建立：

- 主仓噪音只读扫描脚本：`quant/main/tools/repo_noise_report.py`
- 该脚本仅扫描 `quant/main/*.py` 中明显的研究、调参、验证、诊断和归档辅助候选
- 输出只作为候选队列，不移动、不删除、不归档
- 主线隔离目标目录已建立：`core/`、`workflows/`、`agent_tools/`、`research/`、`legacy/`
- 生产资产变更守门脚本已建立：`quant/main/tools/production_asset_gate.py`
- 生产资产 registry 已登记当前 L1/L2/L3/L4/L5/L7 关键主线资产

后续处理要求：

- 候选文件必须由责任智能体认领
- 审计智能体复核是否仍被标准链路、测试、归档或文档引用
- 确认无用后只能进入项目回收站，不能直接物理删除
