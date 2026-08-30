# workflows 目录

用于承载标准主工作流的脚本化模板和入口约定。

当前原则：

- 尽量用脚本保证稳定性，不靠 prompt 手工拼流程。
- 这里放的是工作流模板、契约和入口说明，不放任意研究脚本。
- 业务层脚本仍由各层 owner 维护，工作流层负责把它们按标准顺序串起来。

当前已落地内容：

- `standard_incremental_trading_signal_l1_l8.json`
  - 当前正式 L1-L8 增量加工模板
  - 明确了 owner、层级、标准任务、gates、hard rules
- `standard_layer_handoff_contract.example.json`
  - 当前统一的分层脚本契约示例
  - 用于约束各层 handoff JSON 的统一外壳字段

与之配套的治理入口：

- `quant/main/tools/workflow_monitor_manage.py`
  - `list-templates`
  - `create-from-template`
- `quant/main/tools/validate_workflow_contract.py`
  - 校验某一层 handoff JSON 是否满足统一脚本契约
- `quant/main/tools/build_workflow_handoff_contract.py`
  - 先把现有层报告适配成统一 handoff contract
  - 当前已支持 `L1`、`L2`

推荐用法：

```powershell
D:/work/quant/quant_mcp/.venv/Scripts/python.exe `
  D:/work/quant/quant_mcp/quant/main/tools/workflow_monitor_manage.py `
  list-templates
```

```powershell
D:/work/quant/quant_mcp/.venv/Scripts/python.exe `
  D:/work/quant/quant_mcp/quant/main/tools/workflow_monitor_manage.py `
  create-from-template `
  --template-id standard_incremental_trading_signal_l1_l8 `
  --workflow-id incremental-trading-signal-20260711 `
  --user-goal "补齐 20260711 的 L1-L8 标准增量加工链路"
```

后续要求：

- 新工作流优先先加模板，再让指挥官按模板建 monitor。
- 若某层入口、资产路由或 gate 规则变化，应同步更新模板，而不是只改 prompt 口径。
- 各层现有 JSON 可继续保留自己的业务字段，但向下游交接时应包进统一 handoff contract 外壳，再交给审计或指挥官消费。

当前推荐迁移顺序：

1. 先保留各层现有业务 JSON
2. 用 `build_workflow_handoff_contract.py` 生成统一交接 contract
3. 让审计和指挥官优先消费 contract
4. 再反推各层生产脚本原生输出 contract
