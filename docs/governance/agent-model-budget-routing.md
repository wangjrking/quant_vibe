# GPT-5.6 智能体模型预算硬路由

> Current project rule (owner-directed): every agent defaults to `gpt-5.6-terra`
> with `medium` thinking.
> A non-Terra model is permitted only when the frontend has an explicit manual
> selection for that dispatch. Prompt text, inherited thread state, or a local
> configuration value is not a manual frontend selection.

## 目标

每次收到主人提示词后，指挥官必须先评估提示词复杂度、期望速度和风险，再结合剩余 Token 与剩余时间，在 GPT-5.6 系列中选择最合适的模型和推理强度。

权威配置：

`quant/main/config/agent_model_budget_routing_policy.json`

确定性选择器：

`quant/main/tools/select_agent_model.py`

## 当前固定路由

当前项目所有新分派统一使用 `gpt-5.6-terra`，推理强度固定为 `medium`。复杂度、速度、预算和风险字段继续保留用于审计记录，但在 `fixed_dispatch.enabled=true` 时不再改变默认模型或推理强度。

平台必须实际返回 Terra 身份；仅在提示词或配置中填写 Terra、而运行环境实际显示其他模型时，任务必须 fail-closed。

## Terra-medium 默认硬约束

- 当前及后续所有新分派统一使用 `gpt-5.6-terra` 和 `medium` thinking。
- 仅当前端对该次分派做出显式手动模型选择时，才可覆盖默认值；提示词、线程历史或本地配置均不构成前端覆盖。
- 选择器必须检查配置和运行时返回的模型身份；身份不是所选模型时 fail-closed，禁止降级、自动替换或中途换模。
- 复杂度、风险、速度和预算只用于审计记录和是否阻断；在固定路由启用时不改变默认模型或推理强度。

## 版本职责

| 版本 | 默认职责 |
| --- | --- |
| `gpt-5.6-terra` | 所有未被前端显式覆盖的当前分派、开发、审计和跨层协同；thinking 固定为 `medium` |

禁止路由到 GPT-5.5、GPT-5.4、mini、spark 或其他非 GPT-5.6 模型。

## 每条提示词的前置判断

### 复杂度

- `1`：状态查询、格式化、机械检查、单步明确操作。
- `2`：小范围修改、清晰测试、普通文档或证据整理。
- `3`：标准开发、跨数个文件分析、常规调试和工作流推进。
- `4`：复杂调试、跨层设计、生产候选或重大缺陷整改。
- `5`：正式审计、生产切换、实盘、权限安全、不可逆或系统性决策。

### 期望速度

- `immediate`：约两分钟内给出可用结果。
- `fast`：约十分钟内完成。
- `normal`：正确性优先的常规交付。
- `deep`：允许充分推理和验证。

### 风险

- `low`：只读或容易回滚。
- `normal`：普通代码和研究资产。
- `high`：跨层、复杂整改或生产候选。
- `critical`：正式生产、审计、paper/live、权限、密钥或不可逆动作。

## 默认映射

| 复杂度 | 默认模型 | 默认推理 |
| ---: | --- | --- |
| 1 | Terra | medium |
| 2 | Terra | medium |
| 3 | Terra | medium |
| 4 | Terra | medium |
| 5 | Terra | medium |

所有风险和速度场景默认均使用 Terra medium；风险只决定预算门禁和是否 fail-closed，不触发模型或推理强度替换。

## Token 与时间规则

1. 同时有 Token 和时间比例时，取较小值作为有效余量。
2. 有效余量 `>=50%`：维持 Terra medium。
3. 有效余量 `25%-50%`：维持 Terra medium；不得为节省资源换模。
4. 有效余量 `10%-25%`：维持 Terra medium；无法满足任务资源需求时阻断。
5. 有效余量 `<10%`：停止新任务，保留资源用于验证和收口。
6. 还必须满足复杂度绝对下限；复杂度 1-5 的 Token 下限依次为 400、1200、3500、8000、14000，时间下限依次为 45、150、420、900、1500 秒。
7. 预算字段不可测时，按复杂度和风险默认值执行，并显式记录 unknown，不得假设资源充足。
8. 总预算至少预留 20% 给测试、审计、回滚验证和最终答复。

## 执行硬规则

1. 每次新提示词、线程分派或阶段升级前必须重新选择。
2. 选择器返回 `dispatch` 才允许分派，并显式传入 `selected_model` 和 `thinking`。
3. 返回 `blocked_budget_insufficient` 时必须停止、拆分或等待，不得换弱模型强行继续。
4. 已运行任务不得中途降低模型或推理强度。
5. 主人显式覆盖只对单次分派有效，必须留存原选择、覆盖值、原因和时间。

## 示例

```powershell
D:/work/quant/quant_mcp/.venv/Scripts/python.exe `
  D:/work/quant/quant_mcp/quant/main/tools/select_agent_model.py `
  --complexity 4 `
  --speed normal `
  --risk high `
  --remaining-tokens 50000 `
  --token-budget 100000 `
  --remaining-seconds 5000 `
  --time-budget-seconds 10000
```
