# 线程制智能体管理规范

## 定位

本文档定义本项目如何用 Codex 会话线程管理正式智能体。

目标不是把线程当普通聊天窗口，而是把线程固化为：

- 长期角色实例
- 可追踪执行入口
- 可审计协同节点
- 自动化可绑定对象

本文档不替代以下文件：

- `WORKFLOW.md`
- `quant/main/AGENTS.md`
- `.codex/agents/communication-layer.md`
- `.codex/agent_packages/registry.md`

当规则冲突时，仍按以下优先级执行：

1. 主人明确指令
2. `WORKFLOW.md`
3. `quant/main/AGENTS.md`
4. `.codex/agents/communication-layer.md`
5. 本文档
6. 其他治理说明

## 核心原则

### 一线程一智能体

每个正式智能体只绑定一个正式线程。

- 一个线程只服务一个智能体身份
- 一个智能体的正式协同、回执、审批申请和阶段进展，都应优先回到其绑定线程
- 不得把某个线程临时借给其他智能体承接发言

### 线程是角色实例，不是临时聊天

线程要承载以下稳定属性：

- 智能体身份
- 层级职责
- 固定边界
- 固定汇报格式
- 固定线程标题
- 固定 thread id
- 固定运行态记忆路径

### 指挥官统一调度

线程制管理下：

- 指挥官智能体负责分派、收口、权限判断和风险把控
- 专业智能体负责在自己的线程里执行和回报
- 审计智能体只在审计线程里输出审计结论

### 状态落文件，协同走线程

线程负责可见交互，文件负责长期留痕。

建议分工：

- 线程：任务分派、进度回报、权限申请、完工结论
- 运行态文件：注册表、协同请求、决策、证据索引、agent notes

## 三个管理对象

线程制智能体管理由三类对象共同组成：

### 1. 线程

负责承载智能体的可见对话和实际调度入口。

### 2. 注册表

负责记录每个智能体对应哪个正式线程，以及与线程管理有关的元数据。

主文件：

- `quant/data_file/runtime/agent_memory/agent_thread_registry.json`
- `.codex/agent_packages/registry.md`

### 3. 自动化

负责按时间唤醒线程或独立执行固定任务。

线程制下推荐：

- heartbeat 自动化绑定线程
- cron 自动化绑定工作目录
- 自动化说明回写到智能体文档包或运行态记录

## 线程命名规范

正式线程标题统一使用中文角色名，必要时可带层级前缀。

推荐格式：

- `指挥官智能体`
- `架构师智能体`
- `审计智能体`
- `数据接入智能体`
- `数据整合智能体`
- `MCP智能体`
- `因子智能体`
- `模型智能体`
- `策略智能体`
- `交易智能体`
- `投研智能体`

可选增强格式：

- `L1-数据接入智能体`
- `L4-模型智能体`
- `L5-策略智能体`

但一旦采用，必须在所有正式线程中统一。

## 注册表字段规范

`agent_thread_registry.json` 至少应包含以下字段：

### 顶层字段

- `schema_version`
- `updated_at`
- `rule`
- `thread_policy`
- `threads`

### 单线程记录字段

- `agent_id`
- `agent_name`
- `layer`
- `thread_id`
- `thread_title`
- `thread_status`
- `source`
- `preferred_model`
- `reasoning_policy`
- `default_message_header`
- `active_automation_ids`
- `notes`

建议含义：

- `thread_status`：`active` / `paused` / `archived`
- `preferred_model`：当前项目默认执行模型；除非前端对该次分派显式手动选择，否则必须与模型路由规范一致
- `reasoning_policy`：当前默认 thinking 与前端覆盖边界，例如“默认 medium；仅前端单次显式选择可覆盖”

模型实际分派必须遵守
`quant/main/docs/governance/agent-model-budget-routing.md`。当前默认模型与 thinking
必须与该规范一致；预算和风险只影响是否阻断，不得静默改用其他模型或推理强度。
- `default_message_header`：例如 `会话来自：模型智能体（model-agent）`
- `active_automation_ids`：绑定到该线程的活跃自动化 id 列表

## 正式成员最小档案

每个正式智能体至少要在以下位置能互相对上：

1. `quant/data_file/runtime/agent_memory/agent_thread_registry.json`
2. `.codex/agent_packages/registry.md`
3. `.codex/agents/README.md`

这三处至少要保持以下信息一致：

- 中文名
- 英文 id
- 层级职责
- 绑定线程 id
- 线程标题

## 线程生命周期

### 新建线程

当新增正式智能体时，顺序如下：

1. 创建正式线程
2. 灌入启动角色卡
3. 记录 thread id
4. 更新线程注册表
5. 更新智能体注册表
6. 更新项目说明入口

### 替换线程

如需替换正式线程，不允许静默替换。

至少要做：

1. 记录旧 thread id
2. 指定新 thread id
3. 更新 `agent_thread_registry.json`
4. 更新 `.codex/agent_packages/registry.md`
5. 更新 `.codex/agents/README.md`
6. 检查相关 heartbeat / automation 是否需要改绑

### 归档线程

线程归档前，应确认：

- 是否仍有自动化在唤醒它
- 是否仍是正式线程
- 是否已有替代线程
- 是否已完成注册表更新

## 任务分派规则

### 标准路径

标准分派路径如下：

```text
主人需求
-> 指挥官智能体判断责任归口
-> 指挥官智能体找到目标线程
-> 指挥官智能体在目标线程下发任务
-> 责任智能体在自己的线程回报
-> 如需审计，指挥官再分派到审计线程
-> 指挥官统一收口给主人
```

### 禁止路径

以下方式不合规：

- 在 A 智能体线程里让 B 智能体直接发言
- 借用别的智能体线程做回执
- 只写通信层但声称“已通知目标智能体”
- 跳过指挥官直接把正式任务转派给下游智能体

## 权限与审批流

线程制管理下，权限申请必须在线程内可见提出。

高风险事项包括但不限于：

- 补数据
- 重算因子
- 模型训练
- 生成预测
- 生成正式信号
- 跑回测
- 修改投产参数
- 模拟盘 / 实盘交易交付

标准流程：

1. 专业智能体在自己的线程里发起权限申请
2. 指挥官判断是否进入用户审批
3. 主人审批
4. 指挥官回到责任线程下发授权结果

## 自动化绑定规则

### heartbeat

适用于：

- 交易监控
- 指挥官日报
- 等待固定时间后继续检查

规则：

- heartbeat 应优先绑定到对应正式线程
- 自动化 id 应登记到 `agent_thread_registry.json`
- 自动化说明应与线程身份一致

### cron

适用于：

- 独立工作目录任务
- 不依赖当前线程上下文的固定作业

规则：

- cron 不等于智能体正式线程
- 如 cron 结果要回到某个智能体，应明确回传责任线程

## 消息协议

线程制要长期稳定，必须固定消息协议。

最小协议应包括：

- 抬头：`会话来自：中文智能体名称（agent-id）`
- 同步类型
- 任务编号
- 日期口径
- 证据路径
- 风险/阻塞/下一步

详细模板主归档位置：

- `.codex/agents/communication-layer.md`

## 推荐维护动作

当你调整智能体体系时，优先检查：

1. 正式线程是否存在
2. 线程标题是否规范
3. `agent_thread_registry.json` 是否更新
4. `.codex/agent_packages/registry.md` 是否更新
5. `.codex/agents/README.md` 是否更新
6. 自动化是否仍绑定正确线程
7. 通信层模板是否还能覆盖新场景

## 当前项目建议

本项目当前最适合的线程制管理结构为：

- 1 个指挥官主调度线程
- 10 个正式专业智能体线程
- 1 份 JSON 注册表
- 1 份 Markdown 注册表
- 1 套运行态记忆文件
- 若干 heartbeat / automation 绑定到对应线程

## 关联文档

- `WORKFLOW.md`
- `quant/main/AGENTS.md`
- `.codex/agents/README.md`
- `.codex/agents/communication-layer.md`
- `.codex/agent_packages/registry.md`
- `quant/data_file/runtime/agent_memory/agent_thread_registry.json`
- `quant/main/docs/governance/project-doc-map.md`
