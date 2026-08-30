# 全量数据初始化工作流

## 适用场景

本工作流用于新电脑首次建库，或已获得明确授权的全历史重建。它与日常 `L1-L8` 增量链路完全隔离：不得复用增量 run-id、workflow monitor、staging 目录或“target-date-only”入口。

日常增量只处理一个交易日；全量初始化覆盖已批准的历史日期范围，因此始终属于严格审计变更。

## 流程

1. **F0 预检**：指挥官冻结日期范围和 run plan；确认 Tushare 凭据、官方交易日历、磁盘/内存容量、目标 active 路由和回滚空间。
2. **F1 全量 L1**：数据接入智能体将原始数据写入隔离候选区，完成 source、Parquet、DuckDB 三方对账，以及 `no-BJ` 和一表一 DuckDB 检查。
3. **F2 全量 L2**：数据整合智能体基于 F1 候选生成隔离的 `STOCK_DAILY_DATA`，核验原始价格、显式 qfq、重复键和回滚候选。
4. **F3 全量 L3**：因子智能体生成隔离 features 与成熟 labels。features 与 labels 必须分开管理，标签不得越过成熟日，所有未来列必须为零。
5. **F4 可选预测重放**：只有正式模型和 formal manifest 均不变时，模型智能体才可对候选 L3 重放正式预测；本步骤不训练、不调参、不替换模型。
6. **逐层审计和切换**：每个 active 切换前必须完成 audit-agent 与 architect-agent 的只读复核，并保留原子回滚资产。

## 默认终点

全量初始化默认止于 F4。它不会自动进入 L5-L8，不生成正式买卖信号，不创建交易交付包，也不触发外部发布或交易。

完成全量初始化后，下一次业务运行回到日常增量链路：从最新官方交易日执行 L1，再按 L2 到 L8 推进。

## 关键边界

- 必须有用户明确授权和指挥官冻结的 run plan。
- 全量资产先写候选区，未审计前不得声明为 active。
- 任一阶段失败必须 quarantine 或 rollback，禁止以局部缺口继续推进。
- 模型训练、策略语义变更和真实交易均属于额外工作流，不能夹带在初始化中。

## 与当前本地资产的关系

当前 active L2、L3 feature 和正式预测资产可以覆盖长期历史；但 active L1 的部分辅助表是为日常增量保留的滚动窗口。这是正常的运行态设计，不能被视为全量原始数据铺底证明。

因此全量初始化必须从官方源重新生成 F1 候选，再推进 F2-F4；不得以当前 active L1 的最早日期代替官方全历史覆盖证明。可使用 `tools/check_workflow_asset_alignment.py` 核对当前增量链路的目标日期对齐。

机器可读合同见 [full_history_initialization_policy_v1_20260830.json](../../config/full_history_initialization_policy_v1_20260830.json)。
