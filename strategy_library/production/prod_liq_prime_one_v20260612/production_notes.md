# 流动性精选一号 V1.0 投产备注

更新时间：2026-06-15

本文记录 `prod_liq_prime_one_v20260612` 的投产口径、预测表血缘、信号筛选规则、交易规则、掘金复现结果和当前复跑差异。该文件用于生产交接和后续复现，不构成未来收益承诺。

## 1. 策略身份

- 策略 ID：`prod_liq_prime_one_v20260612`
- 中文名：流动性精选一号
- 生产版本：V1.0
- 掘金策略目录：`D:/work/dfcf/juejin/strategy/ce327750-634e-11f1-b8d7-10ffe0295517`
- 掘金策略脚本：`D:/work/dfcf/juejin/strategy/ce327750-634e-11f1-b8d7-10ffe0295517/main.py`
- 掘金运行名：`open_daily_nomarket_top1_q95_hold5_score095_mh2`
- 当前脚本 SHA256：`8EA26534569CA66806A8819C421849AE0419683D3B9D888A6DF21C4341A24DCE`
- 主要信号文件：`D:/work/quant/quant_mcp/quant/data_file/gm_signals_prod_aligned_4y_liq_enh_top1_q95_amt8e5_turn2_mv1e6_clean_nost_delist.csv`
- 信号文件 SHA256：`2628A6B76412436ED4CBD30A51D658D705CA05C94A952BF83841544E9334E742`
- 评分数据库：`D:/work/quant/quant_mcp/quant/data_file/odb.db`
- 评分表：`stock_predict_data_executable_5d_open_return_prod_aligned_202206_202606`

## 2. 一句话定义

每天收盘后读取 `executable_5d_open_return` 预测分，在全市场前 5% 分数池中做流动性、换手、市值、ST/退市和可交易过滤，选预测分最高的 1 只股票，于下一个交易日开盘附近买入；持有 5 个交易日，或在最少持有 2 个交易日后若预测评分跌破买入评分的 95% 则提前卖出。

## 3. 预测表血缘

当前投产评分表不是一个单一重训模型直接生成的全周期表，而是由三段组件拼接得到：

| 日期区间 | 来源表 | 行数 | 说明 |
|---|---|---:|---|
| 20220606-20240603 | `stock_predict_data_executable_5d_open_return_prod_rebuild_aligned_early_202206_202406` | 370,682 | 早期组件，股票池约 781 支，接近沪深300+中证500口径 |
| 20240604-20260609 | `stock_predict_data_executable_5d_open_return_m018_scores_currentdata_202406_202606` | 867,177 | 后期 m018 组件，接近全 A |
| 20260610-20260612 | `stock_predict_data_executable_5d_open_return_prod_aligned_increment_20260610_20260612` | 15,617 | 增量组件 |

最终表 `stock_predict_data_executable_5d_open_return_prod_aligned_202206_202606` 与可重建表 `stock_predict_data_executable_5d_open_return_prod_aligned_rebuilt_20260614` 全日期逐行一致：

- 交易日：976
- 行数：1,253,476
- 缺失 key：0
- 多出 key：0
- `pred_prob` 最大差异：0
- `pred_prob` 平均差异：0
- 裸 Top1 一致：976 / 976

注意：20240604-20260609 的全 A 候选表 `stock_predict_data_executable_5d_open_return_rolling_qtr_expanding_2y_foldfs120_fast800_alla_light_m018` 与原 m018 组件在交集上分数完全一致，但多出 759 行。裸 Top1 有 7 天不同；不过按本策略真实交易过滤后，218 个信号日全部一致。

不要把 `stock_predict_data_executable_5d_open_return_rolling_4y_alla_light_m018` 当作投产复现表。它是另一张全 A 重训候选表，与投产表交集分数没有逐行一致，筛选后信号也不一致。

## 4. 模型和标签

- 模型类型：XGBoost regression
- 预测标签：`executable_5d_open_return`
- 标签含义：T 日产生预测；T+1 交易日开盘买入；约 T+6 交易日开盘卖出；扣除训练口径成本后的 5 日可执行收益。
- 训练方式：历史归档描述为 rolling / expanding rebuild aligned。
- 训练起点：`20100101`
- 因子筛选：每个训练折内按 Rank IC 选取候选因子，默认 Top120，最终归档有效因子约 83 个。
- 重要限制：旧投产版本没有保存二进制模型 checkpoint，也没有保存完整训练样本 hash、每日 universe hash 和完整环境快照。因此生产复现应以已归档预测表血缘为准，不能用当前数据重训结果冒充原版。

## 5. 股票筛选规则

每个信号日 T 按以下步骤生成 T+1 开盘买入候选：

1. 读取 T 日预测表。
2. 计算当日 `pred_prob` 的全市场 q95 分位数。
3. 保留 `pred_prob >= q95` 的股票。
4. 排除 ST、退市和名称中含退市风险的股票。
5. 排除开盘不可交易股票；买入时若开盘涨停、停牌、开盘价缺失或为 0，则跳过。
6. 流动性和规模过滤：
   - `amount >= 800000`
   - `turnover_rate >= 2.0`
   - `total_mv <= 1000000`
7. 对剩余股票按 `pred_prob` 降序排序。
8. 只取 Top1。

字段单位沿用项目数据库中的 Tushare/本地因子口径，不按手工人民币直觉重新换算。所有阈值都应保持原样，否则不再是本投产版本。

## 6. 买入和仓位规则

- 执行日：T+1 交易日
- 买入时间：`09:31:00`
- 最大持仓数：1
- 目标仓位：0.98
- 初始资金：600,000
- 滑点：0.0015
- 复权：`none`
- 委托方式：开盘价限价单
- 买入限制：开盘涨停不买；停牌、无价格、价格异常不买。
- 已有持仓且没有触发卖出时，不会买入新 Top1。

## 7. 卖出规则

每天先卖后买：

- 卖出时间：`09:30:00`
- 买入时间：`09:31:00`

卖出触发条件：

1. 固定持有期卖出：持仓满 5 个交易日后卖出。
2. 评分回落提前卖出：持仓至少满 2 个交易日后，如果当前决策日评分满足：

```text
current_score <= entry_score * 0.95
```

则提前卖出。

3. 不延长持仓：V1.0 中 `max_holding_days = holding_days = 5`，因此没有继续持仓扩展逻辑。
4. 开盘跌停不卖：若卖出时开盘跌停或不可交易，则跳过卖出。

持仓天数按策略内部交易日索引计算。举例：如果实际在周五 2026-06-12 买入凡拓数创，则周一 2026-06-15 开盘只算持有 1 个交易日，不满足“最少持有 2 个交易日后才允许评分卖出”的条件，因此 0615 不应因评分回落卖出。

## 8. 掘金复现命令

封存复现命令保存在：

`D:/work/quant/quant_mcp/quant/data_file/reports/prod_liq_prime_one_juejin_verify_20260615/prod_liq_prime_one_v1_0_reproduction_5009pct.json`

核心参数：

```text
--strategy-dir D:/work/dfcf/juejin/strategy/ce327750-634e-11f1-b8d7-10ffe0295517
--signal-file D:/work/quant/quant_mcp/quant/data_file/gm_signals_prod_aligned_4y_liq_enh_top1_q95_amt8e5_turn2_mv1e6_clean_nost_delist.csv
--score-db D:/work/quant/quant_mcp/quant/data_file/odb.db
--score-table stock_predict_data_executable_5d_open_return_prod_aligned_202206_202606
--max-positions 1
--holding-days 5
--target-position-pct 0.98
--score-exit-entry-ratio 0.95
--min-holding-days-before-score-exit 2
--score-continue-entry-ratio 1.0
--max-holding-days 5
--backtest-start "2024-06-05 09:00:00"
--backtest-end "2026-06-13 15:30:00"
--backtest-adjust none
--backtest-initial-cash 600000
--backtest-slippage-ratio 0.0015
```

掘金终端必须已登录，且本地服务 `gmterm-serv.exe` 需要监听 `7001-7004` 端口。若 SDK 报 `无法连接到终端服务`，说明回测没有真正开始。

## 9. 掘金验证结果

### 9.1 归档成功结果

日志：

`D:/work/quant/quant_mcp/quant/data_file/reports/prod_liq_prime_one_juejin_verify_20260615/prod_liq_prime_one_h5_score095_mh2_sell0930_buy0931_window20240605.log`

指标：

| 指标 | 结果 |
|---|---:|
| 累计收益 | 5009.93% |
| 年化收益 | 2387.24% |
| 最大回撤 | 27.34% |
| 夏普 | 1.55 |
| 开仓 / 平仓 | 94 / 93 |
| 胜率 | 61.29% |
| Calmar | 87.31 |

### 9.2 当前终端复跑结果

2026-06-15 10:52-10:56，在同一脚本 hash、同一信号文件 hash、同一预测表、同一核心参数下复跑，当前掘金终端稳定得到：

| 指标 | 结果 |
|---|---:|
| 累计收益 | 4471.36% |
| 年化收益 | 2211.44% |
| 最大回撤 | 27.34% |
| 夏普 | 1.45 |
| 开仓 / 平仓 | 94 / 93 |
| 胜率 | 61.29% |
| Calmar | 80.88 |

对比文件：

`D:/work/quant/quant_mcp/quant/data_file/reports/prod_liq_prime_one_juejin_verify_20260615/prod_liq_prime_one_5009_vs_current_rerun_compare.json`

判断：

- 交易次数、胜负次数、胜率和最大回撤完全一致。
- 累计收益、年化收益和夏普降低。
- 这说明信号和主要交易序列大概率一致，差异更可能来自掘金本地行情缓存、成交价数据、复权/除权处理、撮合细节、账户曲线计算口径，或旧 00:19 那次存在未记录的终端环境状态。
- 因此，目前应同时保留两个结果：5009.93% 作为已归档成功日志，4471.36% 作为当前终端可稳定复跑结果。正式对外或继续调参时，应优先以当前可稳定复跑结果为基准，除非能进一步查明 5009.93% 的环境差异。

## 10. 风险和注意事项

- 旧投产预测表可以精确重建，但旧模型本体不可精确重训复现，因为缺少 checkpoint 和完整训练环境快照。
- 早期 20220606-20240603 组件不是全 A 口径，约 781 支股票；后期 20240604-20260609 才接近全 A。
- 本策略交易次数少，且收益高度集中，存在偶然性和样本依赖。
- 当前表现强，不代表未来延续。应继续做分年 OOS、按市场阶段拆解、交易明细复核、滑点敏感性测试。
- 掘金回测结果会受本地数据缓存和终端状态影响。每次正式复跑必须保存日志、策略脚本 hash、信号文件 hash、评分表版本、SDK 版本和终端状态。
- 后续任何新投产策略必须保存：
  - 原始数据快照或数据库 hash
  - 每日 universe 文件
  - 每折训练样本 key 和 hash
  - 每折特征列表和 hash
  - 模型参数 JSON
  - XGBoost booster/model 文件
  - 预测表 hash
  - 信号表 hash
  - 掘金日志和指标 JSON

## 11. AGENTS.md 复现合规清单

依据 `D:/work/quant/quant_mcp/quant/main/AGENTS.md` 第 6 节，本策略目录已经补齐以下复现证据链：

| 要求 | 当前文件 | 状态 |
|---|---|---|
| 策略 manifest | `strategy_manifest.json` | 已有 |
| 中文说明报告 | `report.md`、`production_notes.md` | 已补充 |
| 交易规则 | `trading_rules.json` | 已有 |
| 复现入口 | `reproduction_v1_0.json`、`data_file/reports/prod_liq_prime_one_juejin_verify_20260615/prod_liq_prime_one_v1_0_reproduction_5009pct.json` | 已有 |
| 验证结果 | `validation.json`、`backtests/*.json` | 已补齐 |
| 模型参数 | `model_params.json` | 已有，但历史参数仍有不完整处 |
| 因子列表 | `factors.json` | 已有 |
| 掘金脚本快照 | `code_snapshot/juejin_main.py` | 已补齐 |
| 回测入口快照 | `code_snapshot/run_juejin_signal_backtest.py` | 已补齐 |
| 生产信号快照 | `signals/production_signals.csv`、`signals/signals_latest.csv` | 已补齐 |
| 预测表元数据和 hash | `predictions/prediction_table_meta.json` | 已补齐 |
| 回测日志 | `backtests/gm_backtest_20260615_0019_5009pct.log`、`backtests/gm_backtest_20260615_1054_current_rerun_4471pct.log` | 已补齐 |
| 回测指标 JSON | `backtests/gm_backtest_20260615_0019_5009pct_metrics.json`、`backtests/gm_backtest_20260615_1054_current_rerun_4471pct_metrics.json` | 已补齐 |
| 产物 hash 总表 | `artifact_hashes.json` | 已补齐 |
| Python / SDK 环境 | `environment/python_version.txt`、`environment/python_packages.json`、`environment/sdk_versions.txt`、`environment/juejin_terminal_state_20260615.json` | 已补齐 |
| 训练样本 hash | `datasets/dataset_hashes.json` | 历史缺失，已说明 |
| 模型 checkpoint hash | `models/model_hashes.json` | 历史缺失，已说明 |
| 每折特征筛选明细 | `feature_selection/feature_selection_manifest.json` | 历史缺失，已说明 |

严格复现级别判定：

- **预测表复现：严格复现。** `prod_aligned` 与 `prod_aligned_rebuilt` 全日期逐行 `pred_prob` 差异为 0。
- **信号复现：严格复现。** 生产信号 CSV 已保存 hash，20240604-20260609 的全 A m018 候选表按真实交易过滤后与原投产信号 218/218 一致。
- **掘金回测复现：部分稳定。** 当前终端可稳定复跑 4471.36% 累计收益；5009.93% 有归档成功日志但当前终端未能完全复出，必须保留差异说明。
- **模型重训复现：不能称为严格复现。** 原始模型 checkpoint、训练样本 hash、每日 universe hash 和每折完整特征 IC 证据未归档。后续如果从当前数据重训，只能称为新模型或流程复现，不能冒充此投产原版。

强制约束：

1. 不得用当前数据重训结果覆盖本策略的生产预测表。
2. 不得删除 `backtests/` 中 5009.93% 与 4471.36% 两份日志；两者共同解释当前复现状态。
3. 不得把 `rolling_4y_alla_light_m018` 当作本策略投产预测表。
4. 每次掘金复跑前必须确认 `gmterm-serv.exe` 已监听 `7001-7004`，否则 `GmError status=1001` 代表回测未开始。
5. 每次正式复跑后必须更新或新增回测日志、指标 JSON 和 hash，不允许只口头记录。
