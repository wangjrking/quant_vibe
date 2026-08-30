# 目标交易日增量链路契约

本文件是 2026-08-05 起的标准路由补充，优先级高于历史说明。用户明确提出“增量补齐”时，只能使用目标交易日增量链路，不能因为 L3 旧规程或候选重建模板而切换到全历史入口。

## 唯一路由

`L1 -> L2 -> L3 -> L4 -> L5/L6 -> L7 -> L8`

每一层都必须绑定同一个 `target_trade_date`，执行范围必须是 `target_trade_date_only`，并且 `full_history_rebuild=false`。机器契约由 `quant/main/workflow_route_guard.py` 生成和校验，monitor 从标准模板创建时自动校验；校验失败时不得创建可执行的工作流记录。

标准入口固定为：

- L1：`run_all_a_raw_update.py`
- L2：`integrate_l2_latest_and_recompute_qfq.py`
- L3：`deliver_l3_target_date_duckdb_mainline.py`
- L4：`incremental_formal_l4_duckdb_mainline.py`
- L5/L6：`run_production_tasks.py`
- L7：`build_l7_delivery_package.py`
- L8：`workflow_monitor_manage.py`

以下入口不得出现在本路由中：`rebuild_l2_stock_daily_duckdb_mainline.py`、`rebuild_l3_full_duckdb_mainline.py`、`rebuild_l3_memory_bounded_v2_candidate.py`、`refresh_l3_active_duckdb_full_delivery.py`、`rebuild_factor_data_batched.py`、`gtja_alpha_workflow.py`。

## 启动门

## 日常快速路径

没有 Schema、代码、模型、active 切换或全历史重建时，使用 `routine_target_date_incremental_fast_path_v1`：

- 一个 run-id 和一个总 evidence manifest；
- 启动时一次统一 preflight，后续只做轻量指纹复核；
- L1-L4 保留机器化 handoff 校验，并在 L4 完成后做一次覆盖 L1-L4 的批次审计；不逐层做固定人工审计；
- 不生成 candidate build grant，不重复做 architect review，不重复跑证据包四解析器；
- L7 买入日硬门和 L8 非可执行状态仍然保留。

授权与推进采用 `owner_approved_once_auto_advance_v1`：

- 主人一次明确的目标日增量请求覆盖 L1-L6 日常步骤，不再逐层申请同义 grant；
- 审计通过自动分派下一层；
- 控制线程超时只重发消息，不算业务失败、不把工作流标为 blocked；
- 基础设施类故障允许一次受控整改重试，使用新 attempt id 和空 roots；
- 数据质量、源未就绪、schema 漂移、未来泄漏、未知 writer、资产指纹漂移立即硬停；
- L7 执行、L8 外部发布和 full-history 始终不在该总授权内。

以下情况自动升级到完整治理流程：Schema/代码/模型变化、active 或 registry 切换、全历史重建、生产模型重训、L7 执行授权或外部发布。

创建 monitor 前必须先运行：

```powershell
D:/work/quant/quant_mcp/.venv/Scripts/python.exe D:/work/quant/quant_mcp/quant/main/workflow_route_guard.py --target-trade-date YYYYMMDD
```

实际业务脚本仍由对应专业智能体执行。路由门只负责防止入口错配，不替代 L1-L8 各层的数据质量、资产指纹、买入日硬门和人工确认。审计节奏以 `quant/main/config/incremental_audit_cadence_policy_v1_20260829.json` 为准。

## 失败时的行为

- 发现全量入口、目标日不一致或范围字段漂移：立即停止调度，不读取业务资产，不创建 build workspace。
- 全量重建如确有必要，必须另建明确的 `full_history` 工作流和独立审批，不得复用本增量 monitor、run-id 或证据目录。
- 增量链路的 L3 使用目标日入口；不得把目标日结果包装为全历史候选，也不得把旧全量候选混入本次 L4-L8。
