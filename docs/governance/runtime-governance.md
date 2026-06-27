# runtime 目录治理规则

本文档定义 `quant/data_file/runtime/` 的保留、归档和回收规则。

## 定位

`runtime` 是运行态目录，用于保存智能体记忆、协同记录、监控数据、每日汇报、交易运行留痕、临时复现结果和项目回收站。

它不是源码主目录，也不是长期文档主入口。整理时应优先保护证据链，再降低临时目录噪音。

## 长期保留目录

以下目录默认长期保留，不得按“文件多”直接清理：

| 目录 | 用途 |
| --- | --- |
| `agent_memory/` | 智能体线程注册表、协同请求、决策记录、agent notes |
| `agent_observability/` | 智能体可观测性数据 |
| `agent_workspaces/` | 各智能体运行工作区 |
| `orchestrator_reports/` | 指挥官日报、HTML 报告、工作流监控报告 |
| `recycle_bin/` | 项目内回收站和 30 天观察机制 |
| `trading_agent/` | 交易智能体交付、下单模板、交易运行证据 |

## 阶段归档目录

以下类型可以进入 `runtime/archive/`：

- 已完成且仍有审计或复现价值的 `snapshot_*` 目录。
- 已完成的 `strategy_repro*`、`strategy_snapshot_repro*` 目录。
- 已完成的专项 debug 或 replay 目录。
- 仍被报告引用，但不需要留在 runtime 根目录的证据目录。

归档时必须保留：

- 原路径。
- 归档时间。
- 归档责任智能体。
- 归档原因。
- 是否仍被报告或线程记录引用。

## 回收站候选

以下类型可以进入项目回收站：

- 明确可再生成的缓存。
- 无审计价值的临时测试产物。
- 无引用的 `tmp_*` 目录。
- 过期临时日志目录。

所有回收站动作必须使用：

```powershell
D:\work\quant\quant_mcp\.venv\Scripts\python.exe `
  D:\work\quant\quant_mcp\quant\main\tools\recycle_bin.py `
  move `
  --actor <agent-id> `
  --reason "<中文原因>" `
  <path>
```

不得直接物理删除。

## 禁止事项

- 不得清理 `agent_memory/agent_thread_registry.json`。
- 不得清理当前仍在运行的任务日志。
- 不得移动真实 Token、账号、密钥或未脱敏凭据到报告或公开文档。
- 不得把 runtime 临时目录伪装为正式生产资产。
- 不得因目录名称含 `tmp` 就直接删除，必须先检查引用和证据价值。

## 当前执行口径

第一阶段只执行高置信度缓存和临时测试产物回收。

涉及 `snapshot_*`、`strategy_repro*`、交易运行证据和指挥官日报的目录，先进入整理台账，等待责任智能体和审计智能体复核后再迁移。
