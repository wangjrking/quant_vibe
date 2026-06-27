# 正式资产 GitHub 自动更新机制方案
## 定位

本方案用于设计“正式资产变动后自动同步 GitHub”的机制。当前只定义架构和治理边界，不启用真实自动推送。

## 目标

当生产轨道正式资产发生变动时，自动收集可版本化材料，并在主人批准的发布策略下同步到 GitHub。

## 触发条件

建议触发条件：

- `quant/data_file/asset_registry/production_assets.json` 发生变动。
- 生产 manifest 状态变为 `approved_for_l5` 或 `production_active`。
- `quant/main/strategy_library/` 中投产策略包发生变动。
- 生产任务配置模板发生变动。
- 审计报告确认正式资产变动通过。

## 默认上传范围

默认允许上传：

- manifest。
- 资产注册表。
- 审计报告。
- 策略复现说明。
- 配置模板。
- 哈希摘要。
- 变更说明。
- 不含敏感信息的治理文档。

默认禁止上传：

- 真实 Token、账号、密钥。
- 未脱敏交易账户信息。
- 大型 DB 文件。
- 大型 parquet 分区。
- 大型模型二进制。
- 未审计实验资产。
- 未经主人批准的实盘交付文件。

## 推荐流程

```text
正式资产变动
-> 责任智能体输出 manifest 和证据
-> 审计智能体复核证据
-> 架构师智能体检查 GitHub 发布边界
-> 指挥官智能体确认是否触发发布
-> 主人审批发布方式
-> 自动创建分支
-> 提交可版本化材料
-> 推送 GitHub
-> 创建 PR 或按授权更新指定分支
-> 指挥官汇总 GitHub 链接和变更摘要
```

## 分支策略

建议默认使用独立分支：

```text
asset-update/YYYYMMDD-asset-id
```

默认创建 PR，不直接更新主分支。直接更新主分支必须由主人明确批准。

## 提交内容

建议提交内容：

```text
quant/data_file/asset_registry/
quant/main/config/
quant/main/strategy_library/
quant/main/docs/
quant/data_file/reports/   # 仅提交被确认需要版本化的报告
```

如果资产文件过大，应只提交：

```text
manifest
hash
storage_uri
lineage
audit_report
reproduction_note
```

## 回滚策略

每次发布必须记录：

```text
source_asset_id
target_branch
commit_sha
previous_production_asset_id
rollback_instruction
audit_report
approved_by
```

## 智能体分工

| 智能体 | 职责 |
|---|---|
| 指挥官智能体 | 判断是否触发发布、请求主人审批、汇总结果 |
| 架构师智能体 | 设计发布边界、检查路径和机制 |
| 审计智能体 | 复核证据、敏感信息和误发风险 |
| 数据/因子/模型/策略/交易智能体 | 提供对应层级 manifest、报告和变更摘要 |
| 投研智能体 | 提供需要版本化的研究建议文档 |

## 审批边界

以下动作必须主人审批：

- 首次启用自动推送。
- 允许直接更新主分支。
- 上传大型资产。
- 上传实盘或模拟盘交付文件。
- 上传包含策略参数或交易敏感信息的文件。
- 修改 GitHub 发布目标仓库或分支。

## 后续落地计划

1. 实现只读变更检测脚本。
2. 实现发布候选清单生成脚本。
3. 实现敏感信息扫描。
4. 实现 dry-run 模式。
5. 经主人审批后接入 GitHub push 或 PR 创建。
