# Quant Vibe 多因子量化项目

Quant Vibe 是一个面向 A 股的本地量化研究与信号生成项目，覆盖数据接入、特征加工、模型预测、组合信号和非执行态交易交付。

项目仅在本机 Python 与 DuckDB 环境中运行，不提供云中心或云服务器部署方案。

## 技术架构

```mermaid
flowchart LR
    TS["Tushare 官方数据"] --> L1["L1 原始数据"]
    L1 --> L2["L2 市场底表\n复权与质量校验"]
    L2 --> L3["L3 特征与标签"]
    L3 --> L4["L4 四期限模型\n1D / 3D / 5D / 10D"]
    L4 --> L5["L5 策略信号\n组合构建"]
    L5 --> L6["L6 信号校验\n买入日门控"]
    L6 --> L7["L7 交易交付\nPending-only"]
    L7 --> L8["L8 治理登记\n非执行态"]

    DATA[("data_file/\n本地数据与证据")] --- L1
    DATA --- L2
    DATA --- L3
    DATA --- L4
    DATA --- L5
    CFG["config.json / 环境变量"] --> L1
    CFG --> L4
    CFG --> L5
    GOV["自动门禁与审计"] -. "质量、血缘、回滚" .-> L1
    GOV -.-> L4
    GOV -.-> L6
```

## 快速开始

### 1. 打开项目

```powershell
cd D:\work\quant\quant_mcp\quant\main
```

### 2. 创建 Python 环境

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置数据令牌

推荐通过环境变量提供 Tushare Token：

```powershell
$env:TUSHARE_TOKEN="your_token_here"
```

也可从模板创建本地配置：

```powershell
copy config.example.json config.json
```

## 本地部署与自检

本项目不需要云服务器。完成环境安装和 Token 配置后，在项目根目录执行：

```powershell
cd D:\work\quant\quant_mcp\quant\main
.\.venv\Scripts\python.exe tools\agent_governance_check.py
.\.venv\Scripts\python.exe tools\standard_agent_architecture_check.py
```

两项检查通过后，即可按 L1-L8 工作流运行本地任务。Docker 仅用于本机可复现环境；当前没有 Docker Desktop 也不影响标准流水线。

## Codex 多智能体配置

在 Codex 桌面端打开工作区 `D:\work\quant\quant_mcp`，为下列角色各创建并置顶一个独立任务。任务之间只通过明确交接消息协作，不在其他角色任务中隐式代跑本角色工作。

| 独立任务 | 角色入口 | 职责 |
| --- | --- | --- |
| 指挥官智能体 | `.codex/skills/commander-agent/SKILL.md` | 接收需求、显式分派、汇总状态与裁决下一步 |
| 审计智能体 | `.codex/skills/audit-agent/SKILL.md` | 只读复核关键变更与交接结论 |
| 架构师智能体 | `.codex/skills/architect-agent/SKILL.md` | 审查跨层设计、边界与例外 |
| 数据接入智能体 | `.codex/skills/data-ingestion-agent/SKILL.md` | L1 官方数据接入与原始表质量 |
| 数据整合智能体 | `.codex/skills/data-integration-agent/SKILL.md` | L2 市场底表、复权与回滚 |
| 因子智能体 | `.codex/skills/factor-agent/SKILL.md` | L3 特征与标签 |
| 模型智能体 | `.codex/skills/model-agent/SKILL.md` | L4 训练、预测与模型资产 |
| 策略智能体 | `.codex/skills/strategy-agent/SKILL.md` | L5/L6 信号、组合与验证 |
| 交易智能体 | `.codex/skills/trading-agent/SKILL.md` | L7 pending-only 交付与买入日门控 |
| MCP 智能体 | `.codex/agent_packages/mcp-agent/` | L8 非执行治理登记 |
| 投研智能体 | `.codex/skills/research-agent/SKILL.md` | 研究设计与外部事实整理 |

向每个任务发送的首条配置消息可使用：

```text
你是 Quant Vibe 的【角色名称】。先读取对应角色入口及其 agent package；仅在本角色边界内工作。
所有跨角色工作必须由指挥官在可见任务中显式分派；完成后只回传证据根、结论、唯一阻断和下一步。
默认使用 TERRA，thinking=medium；只有前端明确指定模型时才覆盖。
```

协作规则：指挥官负责显式分派；审计只读复核；各专业角色不替代其他角色执行。日常标准批次按机器门禁运行，出现数据例外、代码或模型变更、路由变更、真实执行时再升级审计。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `README.md` | 项目首页与快速开始 |
| `AGENTS.md` | 多智能体职责、模型路由与协作规则 |
| `docs/governance/` | 工作流、数据契约、治理和部署说明 |
| `config/` | 运行与自动化配置模板 |
| `data_file/` | 本地 DuckDB 数据、报告、信号和运行证据，不提交 |
| `log/`、`logs/` | 本地运行日志，不提交 |
| `juejin_strategies/` | 本地 GM 策略文件，不提交 |
| `tools/` | 校验、审计、切换和运维脚本 |
| `tests/` | 自动化测试 |
| `Dockerfile`、`docker-compose.yml` | 可选的本地容器运行入口 |

## 文档导航

| 主题 | 文档 |
| --- | --- |
| L1-L8 增量工作流 | [latest-incremental-workflow-l1-l8.md](docs/governance/latest-incremental-workflow-l1-l8.md) |
| 数据与路由契约 | [incremental-route-contract.md](docs/governance/incremental-route-contract.md) |
| 多智能体与线程协作 | [thread-based-agent-management.md](docs/governance/thread-based-agent-management.md) |
| 运行与模型治理 | [runtime-governance.md](docs/governance/runtime-governance.md) |
| 项目文档索引 | [project-doc-map.md](docs/governance/project-doc-map.md) |

## 使用边界

- `config.json`、`data_file/`、日志和本地 GM 策略均不提交到 Git。
- 标准生产链路使用受控的 L1-L8 分层资产；历史兼容资产仅可在明确授权下使用。
- 本项目用于研究和策略验证，不构成投资建议。真实交易前必须独立核验数据、成本、风控与执行条件。
