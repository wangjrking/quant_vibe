# 策略验证简化流程

策略研究统一使用以下三步，不再按单个日期、路径或工程错误拆分整改任务：

```text
Build -> Observation -> 一次只读结果审计
```

## 仅保留四条硬约束

1. 开发期与验证期隔离，不偷看验证收益。
2. 数据必须满足 PIT；缺失或冲突不得猜测、伪造或跨日期外推。
3. 观察运行前冻结候选和生产基线，运行中不调参、不扩候选。
4. 研究不得修改生产资产，也不得触发真实交易。

路径、导入、类型、schema、参数传递、临时目录、PID、lease、SHA、manifest
和测试断言均属于 Build 内部工程事项，应在同一任务内修复并重试，不向用户逐项申请。

## 单一验证日历

- 验证日历必须来自一份显式、权威、不可变的开市日历。
- L2、L3、L4 和策略输入只能与该日历比对，不能各自定义日历。
- 禁止把多表交集当成验证日历；交集会掩盖某一层缺少整日数据的问题。
- Build 必须一次性报告所有资产的缺日、多余日、重复键和跨表键差异。
- 独立单日恢复可以作为只读分片与主资产组合消费，无需复制整库；恢复分片必须与主资产日期非重叠、键域并集完整，并且只能通过 `validate_validation_calendar_contract.py` 组合，禁止其他脚本旁路拼接；分片重叠键必须拒绝。
- 所有日期和键完整闭合后，才能开始 Observation。

共享检查入口：

```powershell
D:/work/quant/quant_mcp/.venv/Scripts/python.exe `
  D:/work/quant/quant_mcp/quant/main/tools/validate_validation_calendar_contract.py `
  <descriptor.json> --output <report.json>
```

该检查只验证日历和键域，不计算候选收益、Sharpe、回撤或排名。
