# Active Formal L4 策略调优阶段报告

## 当前结论

本轮继续按最新 active formal L4 生产模型资产调优买入、卖出和次日开盘条件。结论是：当前没有找到满足目标的策略。

目标要求：

- 年化收益率接近或达到 500%
- Sharpe 接近或达到 4
- 最大回撤低于 40%
- 策略平滑、低偶然性、可复现
- 使用最新 L4 formal / DuckDB-only 资产
- 交易执行价格使用未复权裸价

本轮最好的掘金验证候选为：

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| `h5_s10_top1_r950_pctm4p0_stableopen_amt9w_mvany` | 23.39% | 0.650 | 29.79% | 652 / 646 | 47.52% |
| `h10_s10_top1_r970_pctm1p75_stableopen_amt9w_mvany` | 19.75% | 0.731 | 25.87% | 688 / 680 | 46.91% |
| `h10_s10_top1_r970_pctm1p75_stableopen_amt20w_mvany` | 6.87% | 0.412 | 30.74% | 509 / 504 | 44.44% |

均未达标。

## 使用资产

- 1D manifest：`D:/work/quant/quant_mcp/quant/main/config/prediction_manifests/executable_1d_open_return_l4_formal_20260619.json`
- 3D manifest：`D:/work/quant/quant_mcp/quant/main/config/prediction_manifests/executable_3d_open_return_l4_formal_20260617.json`
- 5D manifest：`D:/work/quant/quant_mcp/quant/main/config/prediction_manifests/executable_5d_open_return_l4_formal_20260620.json`
- 10D manifest：`D:/work/quant/quant_mcp/quant/main/config/prediction_manifests/executable_10d_open_return_l4_formal_20260617.json`
- L2 行情：`D:/work/quant/quant_mcp/quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb::STOCK_DAILY_DATA`
- 回测平台：掘金
- 回测复权：`none`
- 回测滑点：`0.003`

## 已验证方向

### 1. 次日开盘缺口过滤

对当前可复跑窄信号做了开盘缺口过滤和仓位缩放。

结果：

- 基线 `baseline_capped`：年化 219.17%，Sharpe 3.774，最大回撤 3.54%
- Sharpe 最高 `gap_m2_to_0p5`：年化 151.90%，Sharpe 4.410，最大回撤 5.92%

判断：

- 开盘缺口过滤能提高 Sharpe，但收益下降明显。
- 这是减少交易换来的，不是收益能力增强。

### 2. Active formal L4 高分宽池，持有 1 日

用 active formal 1D/3D/5D/10D + 未复权执行行情做本地紧凑搜索。

结果：

- 高分一致性、回调、稳定开盘、流动性过滤组合，在持有 1 日口径下整体为负。
- 最好组合仍为负年化。

判断：

- 当前 active formal 高分宽池不支持短持有 1 日策略。

### 3. Active formal L4 高分宽池，持有 5/10 日

本地简化搜索一度显示 10D Top1 有高收益，但最大回撤接近 99%。进一步掘金验证后发现，本地高收益来自错误的重叠仓位简化。

修正为真实梯队仓位：

- 10 日持有：每天新开 9%，最多 10 个持仓
- 5 日持有：每天新开 18%，最多 5 个持仓

掘金结果明显回落，最好年化只有 23.39%。

判断：

- 当前 L4 10D 高分有一定弱正收益，但强度不足。
- 无法通过调持有期直接得到 500% 年化和 Sharpe 4。

### 4. 市场状态过滤

继续测试了中证 2000 市场状态过滤：只在 `index_2000_close` 高于 20 日均线、近 5 日不弱、信号日不大跌的环境中执行。

掘金结果：

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| `h10_s10_top1_r970_pctm175_stable_idxup` | 19.68% | 1.131 | 14.45% | 380 / 374 | 46.26% |
| `h10_s10_top1_r970_pctm175_stable_idxstrong` | 13.25% | 1.043 | 13.59% | 249 / 249 | 45.38% |
| `h10_s10_top2_r950_pctm175_wide_idxup` | 8.28% | 0.634 | 13.68% | 752 / 741 | 45.21% |
| `h5_s10_top1_r950_pctm4_stable_idxup` | 2.97% | 0.170 | 18.27% | 346 / 344 | 42.73% |

判断：

- 市场状态过滤能显著压低回撤。
- Sharpe 有改善，但仍远低于 4。
- 年化收益仍远低于 500% 目标。
- 因此市场过滤可作为风控项，但不能单独解决收益目标。

### 5. 行业强度过滤

继续测试了行业强度过滤：要求候选股所属行业近 5 日或近 20 日收益为正，或近 5 日相对全市场更强。行业字段来自 L2 `STOCK_DAILY_DATA.industry`，行业收益由信号日前可见的 `pct_chg` 聚合得到，不使用未来数据。

掘金结果：

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| `h10_s10_top1_r970_pctm175_ind5rel` | 3.91% | 0.236 | 30.08% | 558 / 555 | 44.50% |
| `h10_s10_top1_r970_pctm175_ind20pos` | 2.53% | 0.174 | 26.88% | 488 / 486 | 44.24% |
| `h10_s10_top1_r970_pctm175_ind5pos` | 1.63% | 0.105 | 25.35% | 524 / 522 | 44.06% |
| `h10_s10_top2_r950_pctm175_wide_ind5pos` | -0.64% | -0.049 | 31.37% | 1134 / 1125 | 43.29% |
| `h5_s10_top1_r950_pctm4_ind5pos` | -6.49% | -0.399 | 48.72% | 444 / 442 | 40.95% |

判断：

- 行业强度过滤没有提高交易胜率。
- 行业过滤后收益显著低于市场状态过滤。
- 当前 active formal L4 高分回调策略的弱点不是简单行业强弱错配，而是个股信号本身交易期望不足。

## 失败原因

1. 当前 active formal L4 宽池的高分信号在交易层面不够强，尤其扣除 0.3% 双边滑点后胜率不足。
2. 旧窄信号池的高收益不能推广到 active formal 宽池。
3. 次日开盘条件更像风控项，不是收益增强项。
4. 持有 5/10 日可以减少短线噪声，但收益强度不足。
5. 本地简化回测必须经过掘金复核，否则容易出现重叠仓位收益被放大的假象。
6. 市场状态过滤能降低回撤，但会同步降低或限制收益弹性，不能把弱正期望信号放大成高收益策略。
7. 行业强度过滤没有改善胜率和收益，说明问题不只是行业环境选择。

## 当前准入判断

不通过。

未通过项：

- 收益目标未达成。
- Sharpe 目标未达成。
- 宽池推广后收益不足。
- 当前仍没有一个 active formal L4 主线下平滑、低偶然性、可复现的高收益策略。

## 证据路径

- 本地持有期搜索脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_compact_hold_search.py`
- 本地持有期搜索结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/compact_hold_local_search_summary.csv`
- 掘金候选脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_hold_candidates.py`
- 掘金候选结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_hold_candidate_results.csv`
- 掘金日志目录：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_logs/`
- 市场状态过滤脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_market_regime_candidates.py`
- 市场状态过滤结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_market_regime_results.csv`
- 市场状态过滤日志：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_market_regime_logs/`
- 行业强度过滤脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_industry_regime_candidates.py`
- 行业强度过滤结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_industry_regime_results.csv`
- 行业强度过滤日志：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_industry_regime_logs/`
- 次日开盘缺口报告：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_next_open_optimization_20260705/next_open_optimization_report.md`

## 下一步建议

继续围绕旧窄信号池微调的价值有限。若目标仍是高收益和低偶然性，需要换策略结构：

1. 重新定义入场场景，例如市场状态过滤、板块强度、支撑形态或缺口支撑形态；
2. 要求候选池在 active formal L4 下自然形成，而不是继承旧历史信号池；
3. 对任何本地高收益候选，必须立即做掘金验证和仓位重叠审计；
4. 将“本地简化收益不能替代掘金结果”列为策略准入强规则。
## 2026-07-05 补充：宽规则买入和评分退出调参

本轮按用户更正后的目标继续调优策略，不调优模型。验证范围仍限定为最新 active formal L4 / DuckDB-only 资产，交易执行价格使用未复权行情，掘金回测滑点为 0.3%。

新增两组验证：

1. `return pocket` 反弹口袋扩展：
   - 原局部收益口袋在本地单笔统计中看起来较好，但掘金组合回测后收益很低。
   - 放宽成交额、市值、TopN 后，交易次数增加，但收益转弱或转负。
   - 最好结果仅为年化 5.26%、Sharpe 0.860、最大回撤 5.50%，且只有 19 次开仓，不能准入。
   - 宽口径版本最高开仓 121 次，但年化为负。

2. active L4 宽规则融合 + 卖出规则调参：
   - 买入侧测试 10D 单因子高分、10D/5D/1D 融合、Top1/Top2/Top3、3/5/10 日持有、回调和次日开盘缺口过滤。
   - 卖出侧测试固定持有、评分衰退 `exit096`、评分衰退 `exit090`。
   - 最好结果为 `bb_h5_top1_10d93_pctm3_gap_wide_amt9w`，三种卖出模式结果相同：
     - 年化：30.69%
     - Sharpe：0.846
     - 最大回撤：31.26%
     - 开仓/平仓：730 / 725
     - 胜率：45.38%
   - 第二梯队为 `bb_h10_top1_10d97_pctm175_gap_stable_amt9w`：
     - 年化：24.81%
     - Sharpe：0.874
     - 最大回撤：27.09%
     - 开仓/平仓：682 / 673
     - 胜率：46.66%
   - 增加 1D 权重到 15% 后，收益明显转负，最大回撤扩大到 60% 以上。
   - Top2/Top3 扩宽后交易次数增加，但年化和 Sharpe 下降，说明当前信号不是“样本太少导致没放大”，而是扩宽后边际候选质量迅速下降。

本轮新增结论：

- 当前 active formal L4 下，10D 高分 + 回调 + 稳定开盘仍是相对最好的方向，但收益上限只在 20%-30% 年化区间，远低于 500% 年化 / Sharpe 4 目标。
- 评分衰退卖出在本轮候选里没有实质改善，原因是评分退出条件没有比固定持有提前产生有效止损/止盈差异。
- 本地局部收益口袋不能作为准入依据，掘金组合回测已经否定其可推广性。
- 当前没有发现可准入、低偶然性、可复现且接近目标收益的策略。

新增证据路径：

- return pocket 掘金脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_pocket_rebound_candidates.py`
- return pocket 掘金结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_pocket_rebound_results.csv`
- return pocket 掘金日志：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_pocket_rebound_logs/`
- 宽规则融合脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_broad_blend_exit_candidates.py`
- 宽规则融合结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_broad_blend_results.csv`
- 宽规则融合日志：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_broad_blend_logs/`

## 2026-07-05 补充：独立卖出规则和轻止损验证

上一轮发现 `exit090` / `exit096` 与固定持有结果完全一致。复核代码后确认原因不是模型无变化，而是研究信号文件中 `min_holding_days_before_score_exit` 被写为 `holding_days + 1`，等于禁止持有期内触发评分退出。因此本轮重新生成 rank-exit 信号，将 `min_holding_days_before_score_exit` 改为 1，并用当前 10D 截面排名作为独立卖出判断。

### 独立 rank 退出

测试口径：

- 持仓票当前 10D 截面 rank 低于阈值则卖出。
- 阈值测试：0.97、0.95、0.90、0.85、0.80、0.70。
- 不依赖历史买入信号分数，只依赖当前评分截面。

结果：

- 最好 rank-exit 候选为 `bb_h5_top1_10d95_pctm4_gap_stable_amt9w / rank_exit=0.70`：
  - 年化：22.75%
  - Sharpe：0.729
  - 最大回撤：26.11%
  - 开仓/平仓：648 / 643
  - 胜率：46.03%
- 对当前宽规则最好候选 `bb_h5_top1_10d93_pctm3_gap_wide_amt9w`，rank-exit 最好只有年化 17.76%、Sharpe 0.593、最大回撤 25.90%。

判断：

- rank-exit 能降低部分回撤，但收益和 Sharpe 明显下降。
- 当前 10D rank 横向退出卖得过早，不能作为收益增强项。
- 它可作为风控研究项，但不能帮助达到本目标。

### 轻止损退出

测试口径：

- 固定持有策略上叠加开盘轻止损。
- 轻止损阈值：3%、5%、8%、10%、15%。

结果：

- 最好轻止损候选为 `bb_h5_top1_10d93_pctm3_gap_wide_amt9w / light_stop=15%`：
  - 年化：30.70%
  - Sharpe：0.842
  - 最大回撤：33.30%
  - 开仓/平仓：765 / 760
  - 胜率：45.66%
- 该结果与固定持有最好版本年化 30.69%、Sharpe 0.846、最大回撤 31.26% 基本持平，但回撤略大。
- 更紧的 3%-10% 止损多数降低年化，未改善整体风险收益比。

判断：

- 轻止损不能显著提高收益，也没有稳定降低回撤。
- 当前策略的主要问题不是卖出规则单点可修复，而是买入信号本身在 active formal L4 下交易期望不足。

新增证据路径：

- rank-exit 脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_rank_exit_candidates.py`
- rank-exit 结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_rank_exit_results.csv`
- rank-exit 日志：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_rank_exit_logs/`
- 轻止损脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_light_stop_candidates.py`
- 轻止损结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_light_stop_results.csv`
- 轻止损日志：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_light_stop_logs/`

本轮更新后的准入判断：

- 未通过收益目标。
- 未通过 Sharpe 目标。
- 回撤低于 40% 的候选存在，但收益和 Sharpe 不足。
- 卖出规则优化没有把策略推近 500% 年化 / Sharpe 4 目标。
- 当前最合理结论仍是：active formal L4 现有交易信号强度不足，继续在同一买入池上微调卖出规则的边际价值很低。

## 2026-07-05 补充：稳健入场结构搜索与反向反弹验证

基于前面“卖出规则不是主瓶颈”的结论，本轮改为搜索新的买入结构。搜索仍只使用最新 active formal L4 / DuckDB-only 派生数据，执行行情使用未复权，掘金回测滑点为 0.3%。

### 本地稳健结构搜索

先用 `active_l4_candidate_dataset_holds_compact.parquet` 做本地筛选，只作为候选缩小，不作为最终收益结论。

搜索维度：

- 评分结构：10D 单独、10D/5D 融合、10D/5D/1D 融合。
- 分数阈值：0.85、0.90、0.93、0.95。
- 信号日跌幅：不限制、-2%、-3%、-4%。
- 次日开盘缺口：不限制、-3% 到 1%、-2% 到 0.5%。
- 成交额：9 万、20 万。
- 市值：不限制、100 亿以下。
- 持有期：1、3、5、10 日。
- TopN：1、2、3。

筛选要求：

- 事件数不少于 300。
- 买入日不少于 250。
- 至少 3 个年份的事件均值为正。

本地筛选前排集中在：

- 10D 高分。
- 信号日下跌约 3%。
- 次日开盘缺口在 -2% 到 0.5% 或 -3% 到 1%。
- Top1。
- 5 日或 10 日持有。

### 掘金验证结果

将本地稳健前排 5 个候选转成掘金信号复跑。

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| `rob_s10_q95_pctm3_gapm2_0p5_h5_top1` | 42.97% | 1.000 | 33.98% | 722 / 717 | 46.86% |
| `rob_s10_q95_pctm3_gapm3_1_h5_top1` | 34.33% | 0.936 | 30.59% | 703 / 698 | 45.56% |
| `rob_s10_q90_pctm2_gapm3_1_h10_top1` | 27.75% | 0.916 | 30.63% | 780 / 769 | 47.07% |
| `rob_s10_q95_pctm3_gapm2_0p5_h10_top1` | 26.20% | 0.848 | 28.38% | 694 / 684 | 47.22% |
| `rob_s10_q95_pctm4_gapm3_1_h10_top1` | 21.93% | 0.798 | 22.60% | 664 / 654 | 48.78% |

本轮最优提升到年化 42.97%，但仍远低于目标 500% 年化和 Sharpe 4。

### 反向/反弹结构验证

另外测试了“10D 不强但 1D 改善”的深跌反弹结构：

- `r10` 低位或不高。
- `r1` 较强。
- 信号日深跌 6%-8%。
- 次日低开或不高开。

掘金结果全部为负，年化约 -15.91% 到 -20.99%，最大回撤约 71.83% 到 86.80%。该方向排除。

### 本轮新增判断

- 当前 active formal L4 下，最有效的买入结构仍是“10D 高分 + 信号日回调 + 次日开盘不过度高开”。
- 反向模型或深跌反弹不是有效方向，风险很高。
- 通过更稳健的结构搜索，本轮掘金年化从约 30.69% 提到 42.97%，但离准入目标仍很远。
- 若目标维持 500% 年化 / Sharpe 4，继续在当前 active L4 分数上做参数微调，预计边际提升有限。

新增证据路径：

- 稳健结构搜索脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_robust_entry_structure_search.py`
- 稳健结构搜索结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/robust_entry_structure_search.csv`
- 稳健结构掘金脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_robust_entry_candidates.py`
- 稳健结构掘金结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_robust_entry_results.csv`
- 稳健结构掘金日志：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_robust_entry_logs/`
- 反向反弹脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_contrarian_rebound_candidates.py`
- 反向反弹结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_contrarian_rebound_results.csv`
- 反向反弹日志：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_contrarian_rebound_logs/`

## 2026-07-05 补充：当前最优邻域调参

本轮继续围绕上一轮最优结构做邻域调参。基础结构为：

- 10D rank >= 0.95。
- 信号日跌幅 <= -3%。
- 次日开盘缺口在 -2% 到 0.5%。
- Top1。
- 成交额 >= 9 万。
- 滑点 0.3%。
- 回测价格不复权。

### 第一组邻域

测试内容：

- 持有期：4、5、6 日。
- 目标仓位：90%、100%、110%。
- 分数阈值：0.93、0.95、0.97。
- 信号日跌幅阈值：-2.5%、-3%、-3.5%。
- 次日开盘缺口边界：-2.5% 到 0.5%、-2% 到 0.5%、-2% 到 1%、-1.5% 到 0.5%。
- Top1 / Top2。

结果前排：

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| `nb_q95_pctm3_gapm2_0p5_h6_top1_t100` | 54.97% | 1.167 | 30.78% | 712 / 708 | 47.88% |
| `nb_q95_pctm3_gapm2_0p5_h5_top1_t100` | 48.84% | 1.056 | 34.88% | 713 / 708 | 46.89% |
| `nb_q97_pctm3_gapm2_0p5_h5_top1_t100` | 46.63% | 1.068 | 34.79% | 660 / 655 | 47.48% |
| `nb_q95_pctm3_gapm25_0p5_h5_top1_t100` | 38.61% | 0.950 | 36.34% | 683 / 678 | 45.43% |
| `nb_q95_pctm3_gapm2_1_h5_top1_t100` | 38.43% | 0.901 | 34.52% | 668 / 663 | 46.46% |

### 第二组小邻域

继续围绕 `h6` 测试：

- 持有期：5、6、7、8 日。
- 目标仓位：95%、100%、105%、110%。

结果：

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| `nb2_q95_pctm3_gapm2_0p5_h6_top1_t100` | 54.97% | 1.167 | 30.78% | 712 / 708 | 47.88% |
| `nb2_q95_pctm3_gapm2_0p5_h6_top1_t105` | 54.97% | 1.167 | 30.78% | 712 / 708 | 47.88% |
| `nb2_q95_pctm3_gapm2_0p5_h6_top1_t110` | 54.97% | 1.167 | 30.78% | 712 / 708 | 47.88% |
| `nb2_q95_pctm3_gapm2_0p5_h6_top1_t95` | 52.57% | 1.170 | 30.11% | 712 / 708 | 47.88% |

判断：

- 当前最优从 42.97% 提升到 54.97%。
- `h6` 明显优于 `h5/h7/h8`。
- 目标仓位从 100% 到 110% 结果一致，说明现金/下单约束已经使实际仓位无法继续放大，不能靠提高目标仓位继续提升收益。
- Top2 明显弱于 Top1，说明第二名候选质量下降明显。
- 本轮仍未达到 500% 年化和 Sharpe 4，因此不能准入。

当前最优研究候选：

- 名称：`nb_q95_pctm3_gapm2_0p5_h6_top1_t100`
- 年化：54.97%
- Sharpe：1.167
- 最大回撤：30.78%
- 开仓/平仓：712 / 708
- 胜率：47.88%

新增证据路径：

- 邻域脚本 1：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_best_neighborhood_candidates.py`
- 邻域结果 1：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_best_neighborhood_results.csv`
- 邻域日志 1：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_best_neighborhood_logs/`
- 邻域脚本 2：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_best_neighborhood2_candidates.py`
- 邻域结果 2：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_best_neighborhood2_results.csv`
- 邻域日志 2：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_best_neighborhood2_logs/`

## 2026-07-05 补充：h6 最优结构叠加市场状态过滤

本轮测试是否可以通过市场状态过滤提高 Sharpe、降低回撤。基础结构仍为当前最优：

- 10D rank >= 0.95。
- 信号日跌幅 <= -3%。
- 次日开盘缺口在 -2% 到 0.5%。
- Top1。
- 持有 6 日。
- 成交额 >= 9 万。
- 滑点 0.3%。
- 回测价格不复权。

市场状态字段来自 L2 `STOCK_DAILY_DATA` 中的 `index_2000_close/open`，只使用信号日及之前可见的指数 5 日、20 日收益和均线，不使用未来信息。

测试结果：

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| `h6_q95_pctm3_gapm2_0p5_idx_day_noncrash` | 30.77% | 0.901 | 29.36% | 618 / 615 | 46.83% |
| `h6_q95_pctm3_gapm2_0p5_idx_nonweak` | 28.86% | 0.877 | 31.18% | 618 / 615 | 47.32% |
| `h6_q95_pctm3_gapm2_0p5_idx_ma5up` | 23.41% | 0.902 | 22.93% | 373 / 371 | 46.09% |
| `h6_q95_pctm3_gapm2_0p5_idxup` | 18.61% | 0.790 | 16.66% | 389 / 386 | 46.37% |
| `h6_q95_pctm3_gapm2_0p5_idxstrong` | 14.59% | 0.833 | 16.55% | 241 / 241 | 44.81% |

判断：

- 市场过滤能降低回撤，但收益下降更明显。
- Sharpe 没有超过未过滤 h6 最优的 1.167。
- 因此市场过滤不是当前目标的有效增益项，只能作为保守风控备选。
- 当前最优仍是 `nb_q95_pctm3_gapm2_0p5_h6_top1_t100`，年化 54.97%、Sharpe 1.167、最大回撤 30.78%。

新增证据路径：

- h6 市场过滤脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_h6_market_regime_candidates.py`
- h6 市场过滤结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_h6_market_regime_results.csv`
- h6 市场过滤日志：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_h6_market_regime_logs/`

## 2026-07-05 补充：h6 评分续持和买入阈值优化

本轮继续围绕当前 h6 最优结构调卖出频率。前面普通轻止损和 rank-exit 都没有改善，本轮改为测试“评分续持”：

- 基础持有 6 日。
- 到第 6 日时，如果当前 10D 预测分数仍不低于入场分数的一定比例，则继续持有。
- 最大持有期测试 8、10、12、15 日。
- 续持阈值测试 0.94、0.96、0.98、0.99、1.00。

### 评分续持结果

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| `h6_cont_0p98_mh12` | 86.43% | 1.268 | 22.41% | 682 / 678 | 49.71% |
| `h6_cont_0p96_mh12` | 82.31% | 1.197 | 24.33% | 678 / 674 | 50.74% |
| `h6_cont_0p98_mh10` | 75.46% | 1.235 | 24.20% | 689 / 685 | 49.05% |
| `h6_cont_0p98_mh8` | 69.66% | 1.254 | 22.73% | 700 / 696 | 48.28% |

判断：

- 评分续持是目前最有效的卖出频率优化。
- 相比固定 h6 的 54.97%，年化提升到 86.43%，Sharpe 从 1.167 提升到 1.268，最大回撤从 30.78% 降到 22.41%。
- 续持阈值 0.98、最大持有 12 日最好。

### 固定续持后的买入阈值微调

固定卖出规则为：

- 基础持有 6 日。
- `score_continue_entry_ratio = 0.98`。
- `max_holding_days = 12`。

继续微调 10D 分数阈值、信号日跌幅和次日开盘缺口。

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| `h6c98_q97_pctm3_gapm2_0p5` | 90.91% | 1.421 | 22.95% | 633 / 629 | 50.56% |
| `h6c98_q95_pctm3_gapm2_0p5` | 86.43% | 1.268 | 22.41% | 682 / 678 | 49.71% |
| `h6c98_q98_pctm3_gapm2_0p5` | 80.44% | 1.421 | 23.01% | 531 / 527 | 51.42% |
| `h6c98_q99_pctm3_gapm2_0p5` | 58.06% | 1.424 | 18.11% | 382 / 378 | 51.59% |

当前最优研究候选更新为：

- 名称：`h6c98_q97_pctm3_gapm2_0p5`
- 规则：
  - 10D rank >= 0.97。
  - 信号日跌幅 <= -3%。
  - 次日开盘缺口在 -2% 到 0.5%。
  - Top1。
  - 基础持有 6 日。
  - 若第 6 日 10D 分数仍 >= 入场分数的 0.98，则续持。
  - 最大持有 12 日。
  - 成交额 >= 9 万。
  - 滑点 0.3%。
  - 回测价格不复权。
- 掘金结果：
  - 年化：90.91%
  - Sharpe：1.421
  - 最大回撤：22.95%
  - 开仓/平仓：633 / 629
  - 胜率：50.56%

准入判断：

- 相比本轮起点已有明显改善，但仍未达到 500% 年化和 Sharpe 4。
- 回撤要求满足。
- 当前策略收益来自 633 次开仓，不是极少交易尖峰；但参数邻域仍显示 q97 最优，q98/q99 年化下降，仍需后续做时间切片和贡献集中度检查。

新增证据路径：

- h6 续持脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_h6_exit_continue_candidates.py`
- h6 续持结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_h6_exit_continue_results.csv`
- h6 续持邻域脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_h6_continue_neighborhood_candidates.py`
- h6 续持邻域结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_h6_continue_neighborhood_results.csv`
- h6 续持买入邻域脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_h6_continue_entry_neighborhood_candidates.py`
- h6 续持买入邻域结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_h6_continue_entry_neighborhood_results.csv`
- q98/q99 补测结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_h6_continue_entry_q98_q99_results.json`

## 2026-07-05 补充：当前最优分段稳定性验证

为验证当前最优 `h6c98_q97_pctm3_gapm2_0p5` 是否平滑、低偶然、可复现，本轮使用同一信号文件和同一策略参数做掘金分段回测。仅改变 `backtest_start` / `backtest_end`，不改变策略规则。

当前最优信号覆盖：

- 信号行数：699。
- 股票数：546。
- `signal_date`：20220606 到 20260701。
- `buy_date`：20220607 到 20260702。

分段掘金结果：

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| full | 90.91% | 1.421 | 22.95% | 633 / 629 | 50.56% |
| 2022H2 | -1.48% | -0.072 | 18.72% | 103 / 97 | 38.14% |
| 2023 | 14.92% | 0.824 | 9.94% | 117 / 114 | 50.00% |
| 2024 | 57.73% | 1.301 | 22.97% | 158 / 155 | 54.84% |
| 2025 | 46.54% | 1.595 | 9.20% | 155 / 149 | 51.68% |
| 2026YTD | 95.80% | 2.312 | 12.95% | 101 / 97 | 54.64% |
| recent60 | 164.88% | 3.281 | 8.86% | 55 / 51 | 58.82% |

判断：

- 当前策略近期显著增强，2024、2025、2026YTD 均为正，recent60 表现最好。
- 但 2022H2 为负，2023 较弱，说明它不是全周期均匀稳定策略。
- 这版比早期候选更可复现、交易次数更多，但仍存在阶段性依赖。
- 因此仍不能通过“平滑、低偶然、可复现并达到 500% 年化 / Sharpe 4”的完整准入目标。

新增证据路径：

- 分段回测脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_h6c98_q97_slices.py`
- 分段回测结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_h6c98_q97_slice_results.csv`
- 分段回测日志：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_h6c98_q97_slice_logs/`
## 2026-07-05 补充：生产信号仓位、买入过滤和卖出频率调优

本轮从当前生产策略 `prod_deepdrop_smallcap_refill_v20260704` 的可复跑信号出发，不再使用旧候选池或 legacy 预测资产。回测统一使用：

- L4 输入：当前生产归档中已验证的 formal L4 DuckDB score 表。
- 行情执行口径：L2 DuckDB 不复权 `open/pre_close`。
- 掘金回测：`run_juejin_signal_backtest.py`。
- 滑点：0.3%。
- 交易方式：次日开盘买入，持仓 1 天为主。

### 仓位和持有频率网格

固定生产全历史信号，只调目标仓位、持仓天数和最大持仓数。

| 候选 | 持仓 | 目标仓位 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 |
|---|---:|---:|---:|---:|---:|---:|
| `prod_sig_h1_pos0700_mp3` | 1 | 70% | 617.00% | 3.163 | 5.86% | 207 / 207 |
| `prod_sig_h1_pos0600_mp3` | 1 | 60% | 384.29% | 3.318 | 5.04% | 207 / 207 |
| `prod_sig_h1_pos0435_mp3` | 1 | 43.5% | 368.25% | 3.462 | 4.22% | 289 / 289 |
| `prod_sig_h2_pos0435_mp6` | 2 | 43.5% | 302.32% | 2.111 | 11.89% | 302 / 302 |

判断：

- 加大单票仓位可以把年化推过 500%，但 Sharpe 下降，未达到 Sharpe 4。
- 延长持有到 2 或 3 天会明显降低 Sharpe，不是有效卖出优化方向。
- 当前生产信号更适合开盘后一日快进快出；拉长持仓会摊薄收益质量。

### 候选池和仓位邻域

固定持仓 1 天，比较多个生产邻域候选池和 50%/60%/70%/80% 仓位。

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| `refill_mv45_deep82_basefirst_h1_pos80` | 1002.93% | 3.065 | 6.68% | 207 / 207 | 70.53% |
| `refill_mv45_deep82_basefirst_h1_pos70` | 633.78% | 3.211 | 5.86% | 207 / 207 | 70.53% |
| `refill_mv45_deep82_basefirst_h1_pos60` | 394.75% | 3.368 | 5.04% | 207 / 207 | 70.53% |
| `refill_mv45_deep82_basefirst_h1_pos50` | 251.41% | 3.541 | 4.21% | 209 / 209 | 70.81% |

判断：

- 高仓位能显著放大年化，但 Sharpe 随仓位提高而下降。
- 不同 `refill_*_basefirst` 候选池在高仓位下结果高度接近，说明实际成交主要由每天首个可买入信号决定。
- 这不是新 alpha，而是仓位放大和首票优先带来的收益放大。

### 次日开盘缺口和可见入场过滤

本轮按用户要求加入次日开盘信息，但严格使用买入日开盘时可见的不复权 `open/pre_close` 计算开盘缺口，不使用未来收盘或未来收益。

较优结果：

| 候选 | 规则摘要 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---|---:|---:|---:|---:|---:|
| `fine_gap_m2_070_pos072` | 开盘缺口 -2% 到 0.7%，仓位 72% | 568.05% | 3.781 | 3.72% | 178 / 178 | 70.79% |
| `fine_gap_m2_050_pos075` | 开盘缺口 -2% 到 0.5%，仓位 75% | 599.46% | 3.738 | 3.87% | 169 / 169 | 72.19% |
| `fine_mv100_pos070` | 总市值 <= 100 亿，仓位 70% | 589.24% | 3.747 | 5.20% | 139 / 139 | 76.98% |
| `fine_mv100_pos065` | 总市值 <= 100 亿，仓位 65% | 478.74% | 3.822 | 4.86% | 139 / 139 | 76.98% |

判断：

- `fine_gap_m2_070_pos072` 是目前最接近目标的研究候选：年化超过 500%，回撤远低于 40%，但 Sharpe 仍低于 4。
- `fine_mv100_pos065` Sharpe 最高，为 3.822，但年化不到 500%。
- 本轮没有找到同时满足“年化 >= 500%、Sharpe >= 4、最大回撤 < 40%”的组合。

### 分段稳定性验证

对当前最接近目标的 `fine_gap_m2_070_pos072` 做掘金分段复跑：

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 胜率 |
|---|---:|---:|---:|---:|---:|
| full | 568.05% | 3.781 | 3.72% | 178 / 178 | 70.79% |
| 2022H2 | 0.00% | 0.000 | 0.00% | 0 / 0 | - |
| 2023 | 1.68% | 0.307 | 0.00% | 2 / 2 | 50.00% |
| 2024 | 583.94% | 5.408 | 3.72% | 106 / 106 | 71.70% |
| 2025 | 62.64% | 2.880 | 1.79% | 45 / 45 | 71.11% |
| 2026YTD | 97.08% | 4.717 | 2.88% | 25 / 25 | 72.00% |
| recent60_calendar | 133.41% | 5.531 | 2.85% | 15 / 15 | 73.33% |

信号覆盖：

- `fine_gap_m2_070_pos072`：246 行，178 个信号日，178 个买入日，208 只股票。
- 年份分布：2023 年 3 行，2024 年 139 行，2025 年 66 行，2026 年 38 行。

准入判断：

- 收益目标：接近且部分候选达到。
- 回撤目标：达到。
- Sharpe 目标：未达到 4。
- 平滑低偶然：未通过。2024 年贡献明显偏重，2023 样本和收益过弱，阶段稳定性不足。
- 可复现：当前研究回测具备脚本、信号、日志和结果 CSV，可复跑，但尚未进入生产归档。

当前最接近目标的研究候选：

- 名称：`fine_gap_m2_070_pos072`
- 状态：`research_only`
- 规则：生产深跌小市值补位信号 + 买入日不复权开盘缺口在 -2% 到 0.7% + 目标仓位 72% + 持仓 1 天。
- 掘金结果：年化 568.05%，Sharpe 3.781，最大回撤 3.72%。
- 不通过准入原因：Sharpe 未达 4，且分段稳定性不足。

证据路径：

- 仓位/持有频率脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_production_signal_exit_position_grid.py`
- 仓位/持有频率结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_production_signal_exit_position_grid_results.csv`
- 候选池仓位脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_production_pool_position_grid.py`
- 候选池仓位结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_production_pool_position_grid_results.csv`
- 入场过滤脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_production_signal_entry_filter_grid.py`
- 入场过滤结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_production_signal_entry_filter_grid_results.csv`
- 细邻域脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_production_entry_fine_grid.py`
- 细邻域结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_production_entry_fine_grid_results.csv`
- 分段脚本：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/run_juejin_fine_gap_m2_070_pos072_slices.py`
- 分段结果：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_fine_gap_m2_070_pos072_slice_results.csv`
## 2026-07-06 补充：买入日开盘与动态仓位继续优化

本轮继续使用当前生产策略 `prod_deepdrop_smallcap_refill_v20260704` 的正式生产候选池和 active formal L4 DuckDB 资产，不使用旧历史信号池或 legacy SQLite 输入。掘金回测继续使用不复权真实执行价格，`backtest_adjust=none`，统一滑点 `0.003`。

### 新增测试范围

1. 组合过滤网格：成交额门槛、买入日不复权开盘缺口、换手率、信号日跌幅、市值上限。
2. 动态仓位网格：不直接删除弱信号，而是按可执行特征给不同目标仓位。
3. 换手动态仓位细网格：围绕“换手率 >= 4 且买入日开盘缺口不过高”做高低仓位配比。

### 关键结果

| 类型 | 最优候选 | 年化 | Sharpe | 最大回撤 | 结论 |
|---|---|---:|---:|---:|---|
| 组合过滤 | `combo_turn4_gap_m2_070_pos0p75` | 468.51% | 3.848 | 4.06% | 未达到 500% 年化和 Sharpe 4 |
| 动态仓位 | `dyn_gap_m2_070_turn4_hi78_lo435` | 531.35% | 3.866 | 4.36% | 年化达标，Sharpe 未达标 |
| 动态仓位 | `dyn_gap_m2_070_twofactor_hi78_mid62_lo435` | 572.29% | 3.794 | 3.84% | 年化达标，Sharpe 未达标 |
| 换手细网格 | `turnfine_gapm2_0p5_hi0p85_lo0p5` | 652.07% | 3.779 | 4.07% | 年化高，Sharpe 未达标 |
| 换手细网格 | `turnfine_gapm2_0p7_hi0p8_lo0p435` | 565.97% | 3.845 | 4.38% | 较均衡，Sharpe 未达标 |

### 当前判断

本轮没有发现同时满足 `年化 >= 500%`、`Sharpe >= 4`、`最大回撤 < 40%` 的候选。有效方向是：

- 买入日不复权开盘缺口控制在 `[-2%, 0.5%~0.7%]`；
- 换手率较高的信号可以承担更高仓位；
- 弱信号降仓比硬删除更平滑，但仍不能把 Sharpe 推过 4；
- 年化和 Sharpe 在当前候选池内仍存在明显冲突，继续单纯提高仓位会提高年化但压低 Sharpe。

### 证据路径

- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_production_entry_combo_grid_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_production_dynamic_position_grid_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_production_turnover_dynamic_fine_partial_results.csv`

## 2026-07-06 补充：换手动态仓位细网格补齐与次日开盘诊断验证

上轮 `turnfine` 细网格因超时只解析到 64 条日志。本轮将脚本改为可续跑、逐条落盘，并补齐 75 组掘金回测结果。

### 换手动态仓位细网格完整结果

完整细网格没有出现同时满足 `年化 >= 500%`、`Sharpe >= 4`、`最大回撤 < 40%` 的候选。

| 候选 | 年化 | Sharpe | 最大回撤 | 说明 |
|---|---:|---:|---:|---|
| `turnfine_gapm2_0p7_hi0p85_lo0p5` | 702.43% | 3.783 | 4.08% | 年化最高，Sharpe 未达 4 |
| `turnfine_gapm2_0p7_hi0p78_lo0p5` | 564.67% | 3.853 | 3.80% | 年化达标，Sharpe 仍未达 4 |
| `turnfine_gapm2_0p7_hi0p76_lo0p435` | 498.87% | 3.885 | 4.34% | Sharpe 较高，但年化略低于 500% |

结论：在当前生产候选池内，`换手率 >= 4`、买入日不复权开盘缺口 `[-2%, 0.5%~0.7%]`、强信号高仓位/弱信号降仓是目前最有效的局部结构，但仍不能把 Sharpe 推到 4。

### 次日开盘收益诊断验证

本轮使用次日开盘收益只做研究诊断，不把未来收益写入信号规则。诊断发现若按单笔次日开盘收益看，`pred_prob` 或 `pred_10d` 过高的一段反而不占优，因此测试了：

- `pred_prob <= 0.998`；
- `pred_10d <= 0.998`；
- `amount >= 200000`；
- `turnover_rate >= 4`；
- 买入日不复权开盘缺口不高于 `0.3%/0.5%/0.7%`。

掘金复跑结果显示该方向无效：最好的 `nog_pred998_turn4_gap03_pos0p85` 只有年化 `364.82%`、Sharpe `3.452`、最大回撤 `4.51%`，明显弱于当前 `turnfine` 前沿候选。因此，不能把“排除超高预测分”作为后续准入规则。

### 当前有效前沿

| 目标侧重 | 当前前沿候选 | 年化 | Sharpe | 最大回撤 |
|---|---|---:|---:|---:|
| 年化优先 | `turnfine_gapm2_0p7_hi0p85_lo0p5` | 702.43% | 3.783 | 4.08% |
| 均衡优先 | `turnfine_gapm2_0p7_hi0p78_lo0p5` | 564.67% | 3.853 | 3.80% |
| Sharpe 接近优先 | `turnfine_gapm2_0p7_hi0p76_lo0p435` | 498.87% | 3.885 | 4.34% |

以上均未通过完整目标准入，因为没有达到 `年化 >= 500%` 且 `Sharpe >= 4` 的组合要求。

### 新增证据路径

- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_production_turnover_dynamic_fine_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_production_turnover_dynamic_fine_results.json`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/next_open_feature_diagnostic_candidates.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_nextopen_guided_filters_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_nextopen_guided_filters_results.json`

## 2026-07-06 补充：卖出频率和持有结构测试

本轮围绕当前最接近的两个候选：

- `turnfine_gapm2_0p7_hi0p78_lo0p5`：年化 `564.67%`，Sharpe `3.853`，最大回撤 `3.80%`；
- `turnfine_gapm2_0p7_hi0p76_lo0p435`：年化 `498.87%`，Sharpe `3.885`，最大回撤 `4.34%`。

测试方式是复制同一信号文件，只调整 CSV 内的卖出和持有字段，避免 runner 参数与信号内字段不一致：

- `holding_days`；
- `max_holding_days`；
- `score_continue_entry_ratio`；
- `score_exit_entry_ratio`；
- `min_holding_days_before_score_exit`。

### 结果

| 方向 | 代表结果 | 年化 | Sharpe | 最大回撤 | 结论 |
|---|---|---:|---:|---:|---|
| 保持 1 日持有 | `sf_bal_h1_*` | 564.67% | 3.853 | 3.80% | 与原候选一致 |
| 延长到 2 日持有 | `sf_bal_h2_*` | 568.34% | 2.714 | 11.09% | Sharpe 明显恶化 |
| Sharpe 候选 1 日持有 | `sf_shp_h1_*` | 498.87% | 3.885 | 4.34% | 接近但年化略低 |

### 判断

当前策略族的收益主要来自短周期的“买入日开盘到次日开盘”结构。延长持有期并没有改善 Sharpe，反而明显放大波动和回撤。不同 `score_continue_entry_ratio`、`score_exit_entry_ratio` 在 1 日持有结构下几乎不产生差异，说明这套信号当前的核心卖出频率就是 T+1 开盘退出，分数衰退规则不是主要收益来源。

因此，继续优化卖出规则的方向不应是简单延长持仓，而应重新设计“独立卖出评分”或引入更多当日可观测风险状态；在当前代码和信号结构下，单纯调 `holding_days/max_holding_days/score_continue_entry_ratio` 无法把 Sharpe 推到 4。

### 新增证据路径

- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_sell_frequency_grid_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_sell_frequency_grid_results.json`

## 2026-07-06 补充：多信号组合与仓位封顶验证

单一买入过滤和卖出频率调参到达瓶颈后，本轮测试多信号组合。组合对象来自当前 active formal L4 / DuckDB-only 口径下已经掘金复跑过的研究候选：

- `turn_bal`：`turnfine_gapm2_0p7_hi0p78_lo0p5`，偏高年化、换手活跃、开盘缺口控制；
- `mv100_065`：`fine_mv100_pos065`，偏小市值上限和高 Sharpe；
- `combo_turn70`：高换手组合候选。

组合规则：

- 同一 `buy_date + stock_code` 去重；
- 每个买入日最多 3 行；
- 保留 1 日持有、次日开盘退出结构；
- 不使用旧历史信号池、不使用 legacy SQLite 或 research-only L4；
- 掘金回测使用不复权真实执行价格，`backtest_adjust=none`，滑点 `0.003`。

### 多信号组合结果

| 候选 | 年化 | Sharpe | 最大回撤 | 结论 |
|---|---:|---:|---:|---|
| `blend_three_bal_mv_combo` | 414.52% | 4.476 | 3.92% | Sharpe 达标，年化不足 |
| `blend3_scale1p1` | 541.13% | 4.380 | 4.31% | 三项指标达标，但原始日目标仓位最高约 148.5% |

随后对 `blend3_scale1p1` 做日目标仓位封顶，避免目标仓位超过正常账户可解释范围。

### 仓位封顶结果

| 候选 | 日目标仓位上限 | 年化 | Sharpe | 最大回撤 | 结论 |
|---|---:|---:|---:|---:|---|
| `blend3_scale1p1_cap0p90` | 90% | 494.82% | 4.432 | 3.83% | 年化略低于 500 |
| `blend3_scale1p1_cap0p98` | 98% | 541.24% | 4.394 | 4.26% | 三项指标达标，仓位约束可解释 |
| `blend3_scale1p1_cap1p20` | 120% | 509.57% | 4.387 | 4.31% | 指标达标但目标仓位上限偏高 |

当前指标最优可解释候选为：

`blend3_scale1p1_cap0p98`

核心指标：

- 年化：`541.24%`
- Sharpe：`4.394`
- 最大回撤：`4.26%`
- 开仓/平仓：`269 / 269`
- 胜率：约 `70.26%`
- 信号行数：`287`
- 买入日：`195`
- 股票数：`237`
- 每日最多信号数：`3`
- 日目标仓位上限：`98%`
- 平均日目标仓位：约 `64.16%`
- BJ 命中：`0`
- ST/风险警示命中：`0`

### 分段结果

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓 |
|---|---:|---:|---:|---:|
| full repeat | 541.24% | 4.394 | 4.26% | 269 |
| 2022H2 | 0.00% | 0.000 | 0.00% | 0 |
| 2023 | 2.52% | 0.451 | 2.34% | 5 |
| 2024 | 572.66% | 6.185 | 4.26% | 150 |
| 2025 | 64.74% | 3.647 | 3.31% | 74 |
| 2026YTD | 80.61% | 5.003 | 1.74% | 40 |
| recent60 | 96.63% | 5.773 | 1.74% | 15 |

### 准入判断

`blend3_scale1p1_cap0p98` 已通过硬指标层面的研究目标：

- 年化 `> 500%`；
- Sharpe `> 4`；
- 最大回撤 `< 40%`；
- 日目标仓位已封顶到 `98%`；
- 无 BJ、无 ST/风险警示；
- 信号覆盖较单一尖峰候选更宽。

但它仍未完全通过“低偶然/平滑”强准入：

- 2024 年贡献过强；
- 2023 几乎没有有效交易；
- 2025/2026 为正收益，但年化明显低于全周期；
- 当前还需要做贡献集中度、去除最大股票/最大月份/最大 1% 交易后的复测。

因此当前状态应标记为：

**研究指标达标候选，待稳健性准入复核；不能直接发布生产。**

### 新增证据路径

- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_signal_blend_grid_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_signal_blend_scale_grid_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_blend3_scale1p1_exposure_cap_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_blend3_scale1p1_cap0p98_slice_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_blend3_scale1p1_exposure_cap_signals/blend3_scale1p1_cap0p98.csv`

## 2026-07-06 补充：贡献集中度剔除复核

对 `blend3_scale1p1_cap0p98` 继续做脆弱性检查。本轮先用不复权开盘到次日开盘的代理收益识别最大贡献对象，再生成剔除版信号，用掘金重新跑完整资金路径。

代理贡献诊断：

- 最大贡献股票：`300254.SZ`
- 最大贡献月份：`202403`
- 最大 1% 交易阈值：`proxy_weighted_ret >= 0.06916090750293204`

掘金剔除复跑结果：

| 剔除场景 | 年化 | Sharpe | 最大回撤 | 开仓/平仓 | 判断 |
|---|---:|---:|---:|---:|---|
| 原始 `blend3_scale1p1_cap0p98` | 541.24% | 4.394 | 4.26% | 269 / 269 | 硬指标达标 |
| 去掉最大贡献股票 `300254.SZ` | 469.37% | 4.359 | 4.26% | 286 / 286 | 年化跌破 500 |
| 去掉最大贡献月份 `202403` | 410.35% | 4.308 | 4.26% | 267 / 267 | 年化跌破 500 |
| 去掉最大 1% 代理贡献交易 | 402.61% | 4.300 | 4.26% | 284 / 284 | 年化跌破 500 |
| 同时去掉最大股票和最大月份 | 354.76% | 4.260 | 4.26% | 266 / 266 | 年化明显跌破 500 |

### 结论

`blend3_scale1p1_cap0p98` 的 Sharpe 质量在剔除后仍保持 `>4`，说明信号组合本身不是完全无效；但年化 `500%+` 对最大股票、最大月份和顶部交易贡献敏感。按当前准入标准，它仍应标记为：

**硬指标达标但贡献集中度未过强准入，暂不建议发布生产。**

下一步若继续优化，应围绕“保持 Sharpe >4 的同时扩大非 2024/非顶部贡献来源”进行，而不是继续提高仓位。仓位提高会强化贡献集中度风险。

### 新增证据路径

- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/blend3_cap0p98_fragility_diagnostics.json`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_blend3_cap0p98_fragility_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_blend3_cap0p98_fragility_results.json`

## 2026-07-06 补充：扩展组合与补位组合尝试

贡献集中度复核显示 `blend3_scale1p1_cap0p98` 的年化 `500%+` 对最大股票、最大月份和最大 1% 交易敏感。本轮尝试两类改进：

1. 扩展组合：在原组合基础上加入高收益但不同覆盖的 `mv100_080`、`turn_hi85` 等来源，并保持每日最多 3 票、日目标仓位封顶 `98%`。
2. 补位组合：保留原 `blend3_scale1p1_cap0p98` 主信号，只在空位中加入低仓位补位，避免稀释主信号。

### 扩展组合结果

| 候选 | 年化 | Sharpe | 最大回撤 | 结论 |
|---|---:|---:|---:|---|
| `exp_hi85_mv80_combo_cap98` | 460.95% | 4.474 | 3.75% | Sharpe 高，但年化不足 |
| `exp_hi85_mv65_combo_cap98` | 436.79% | 4.472 | 3.92% | 年化不足 |
| `exp_four_bal_hi_mv80_combo_cap98` | 208.64% | 4.696 | 3.11% | 年化明显不足 |

扩展组合提高了 Sharpe，但没有维持 `500%+` 年化，说明增加更多来源会稀释主收益，而不是补足非集中贡献。

### 补位组合结果

补位组合包括：

- `refill_primary_mv80_010/015`
- `refill_primary_turnhi_010/015`
- `refill_primary_lowgap_010/015`

所有补位组合的掘金指标均与原 `blend3_scale1p1_cap0p98` 完全一致：

- 年化：`541.24%`
- Sharpe：`4.394`
- 最大回撤：`4.26%`

原因是原主信号在多数有效买入日已经占据 top3 或补位信号与主信号重复，新增补位没有实际改变资金路径。

### 当前判断

本轮未解决贡献集中度问题。当前最强可解释候选仍是：

`blend3_scale1p1_cap0p98`

但准入状态不变：

**硬指标达标，贡献集中度未过强准入，仍不建议发布生产。**

下一步若继续，应避免简单加仓或简单补位，重点应放在：

- 构造在 2025/2026 独立有效的信号源；
- 降低 2024 强月份占比；
- 或建立分市场状态的稳定信号切换规则，且不能使用月份/年份硬编码。

### 新增证据路径

- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_blend_expanded_cap_grid_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_blend_refill_cap_grid_results.csv`

## 2026-07-06 补充：晚近信号源融合与 cap91 复核

### 晚近信号源融合

基于 `candidate_year_proxy_scan.csv` 选取 2025/2026 代理收益更强的信号源，测试将其与当前硬指标最优 `blend3_scale1p1_cap0p98` 融合。所有版本均继续使用掘金回测、每日最多 3 只、日目标仓位封顶 98%、不复权执行价格。

结果显示，晚近信号源直接参与每日 top3 会明显稀释主信号质量：

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓 |
|---|---:|---:|---:|---:|
| `late_blend_orig70_hybrid20` | 658.59% | 2.999 | 5.89% | 473 |
| `late_blend_hybrid45_orig35` | 215.49% | 2.074 | 18.38% | 473 |
| `late_blend_entry35_hybrid30_orig25` | 176.98% | 2.297 | 11.77% | 473 |
| `late_blend_hybrid35_low30_mv25` | 154.96% | 2.036 | 19.57% | 475 |

判断：晚近源虽然增加了交易次数和覆盖，但收益质量下降，不能作为当前目标的改进方向。

### 仅无信号日补位

进一步测试更保守的补位方式：只有当 `blend3` 当日完全没有信号时，才用 `hybrid_s50_80`、`entry_base80` 或 `prod_sig70` 补位。

最好的补位版本仍未达标：

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓 |
|---|---:|---:|---:|---:|
| `nosig_refill_entry_base80_p0p435` | 447.44% | 3.603 | 4.25% | 300 |
| `nosig_refill_entry_base80_p0p1` | 408.68% | 3.714 | 4.25% | 300 |
| `nosig_refill_hybrid_s50_80_p0p1` | 405.20% | 3.647 | 4.25% | 308 |

判断：补位能够扩大买入日，但新增交易的边际质量不够，反而拉低 Sharpe 和年化，不能解决“平滑且达标”的问题。

### cap 细网格

对 `blend3_scale1p1` 做日目标仓位封顶 `0.91-0.99` 的细网格验证。该测试不改变选股，只改变每日目标仓位封顶。

| 候选 | cap | 年化 | Sharpe | 最大回撤 | 开仓 |
|---|---:|---:|---:|---:|---:|
| `blend3_scale1p1_cap91` | 0.91 | 500.35% | 4.427 | 3.88% | 287 |
| `blend3_scale1p1_cap92` | 0.92 | 505.79% | 4.423 | 3.94% | 287 |
| `blend3_scale1p1_cap95` | 0.95 | 523.19% | 4.409 | 4.10% | 287 |
| `blend3_scale1p1_cap98` | 0.98 | 541.24% | 4.394 | 4.26% | 287 |
| `blend3_scale1p1_cap99` | 0.99 | 547.19% | 4.390 | 4.31% | 287 |

`cap91` 是当前更稳的硬指标候选：年化刚过 500、Sharpe 最高、回撤最低。但它没有改变信号集合，所以仍需要贡献集中度复核。

### cap91 分段和贡献剥离

`blend3_scale1p1_cap91` 分段结果：

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓 |
|---|---:|---:|---:|---:|
| 全周期 | 500.35% | 4.427 | 3.88% | 287 |
| 2023 | 2.52% | 0.451 | 2.12% | 5 |
| 2024 | 537.64% | 6.244 | 3.88% | 161 |
| 2025 | 62.49% | 3.637 | 1.60% | 78 |
| 2026YTD | 79.16% | 5.013 | 1.37% | 43 |
| recent60 | 95.62% | 5.753 | 1.08% | 20 |

贡献诊断：

- 最大贡献股票：`300254.SZ`
- 最大贡献月份：`202406`
- 最大 1% 代理贡献阈值：`0.06838620394561985`

掘金剥离复跑：

| 剥离场景 | 年化 | Sharpe | 最大回撤 | 判断 |
|---|---:|---:|---:|---|
| 去掉最大贡献股票 | 433.73% | 4.390 | 3.88% | 年化跌破 500 |
| 去掉最大贡献月份 | 379.69% | 4.319 | 3.88% | 年化跌破 500 |
| 去掉最大 1% 交易 | 374.00% | 4.330 | 3.88% | 年化跌破 500 |
| 去掉最大股票和最大月份 | 328.03% | 4.274 | 3.88% | 年化明显跌破 500 |

### 当前结论

`blend3_scale1p1_cap91` 是当前更稳的硬指标版本，优于 `cap98` 的地方是 Sharpe 更高、回撤更低、仓位更保守；但它没有解决根本问题：收益仍高度依赖 2024 强阶段和头部贡献。

因此当前状态更新为：

**硬指标最优且更稳的研究候选：`blend3_scale1p1_cap91`。**

**准入状态：硬指标通过，但贡献集中度和年度稳定性仍未过强准入；暂不建议发布生产。**

下一步优化方向不应继续简单补位或加仓，而应重新构造 2025/2026 也有独立正期望的信号源，或者建立基于可观测市场状态的切换规则。当前测试显示，简单扩大覆盖会明显降低 Sharpe。

### 新增证据路径

- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_late_source_blend_grid_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_no_signal_day_refill_grid_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_blend3_cap_fine_grid_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_blend3_cap91_slice_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_blend3_cap91_fragility_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/blend3_cap91_fragility_diagnostics.json`

## 2026-07-06 补充：基于可观测特征的仓位微调

### 诊断依据

对 `blend3_scale1p1_cap91` 的信号做不复权开盘到次日开盘代理收益分桶，发现几个可观测特征：

- 信号日跌幅越深，次日开盘收益越好；
- 买入日低开比平开/高开更好；
- 高换手、高成交额区间收益更好；
- 极高 `pred_10d` 并不更好，过高分位反而收益偏弱。

这些变量都属于信号日或买入开盘可观测字段，没有使用未来标签，也没有使用日期硬编码。

### 特征调仓实验

本轮不删除信号，只调整目标仓位：

- 弱信号：`pct_chg >= -2.5%` 或 `buy_open_gap_raw_pct >= 0`，仓位乘 `0.85`；
- 强信号：`pct_chg <= -5.0%` 且 `buy_open_gap_raw_pct <= -0.8%` 且 `turnover_rate >= 4.5`，仓位乘 `1.05`；
- 日目标仓位仍封顶 `91%`；
- 掘金回测，执行价格不复权。

当前最优特征调仓候选：

| 候选 | 年化 | Sharpe | 最大回撤 | 开仓 | 平均日目标仓位 |
|---|---:|---:|---:|---:|---:|
| `fw_soft_all_weak85_strong105` | 501.69% | 4.442 | 3.83% | 287 | 61.11% |

相对 `cap91`：

| 版本 | 年化 | Sharpe | 最大回撤 |
|---|---:|---:|---:|
| `blend3_scale1p1_cap91` | 500.35% | 4.427 | 3.88% |
| `fw_soft_all_weak85_strong105` | 501.69% | 4.442 | 3.83% |

该版本小幅改善了硬指标，但改善幅度有限。

### 分段结果

| 切片 | 年化 | Sharpe | 最大回撤 | 开仓 |
|---|---:|---:|---:|---:|
| 全周期 | 501.69% | 4.442 | 3.83% | 287 |
| 2023 | 2.67% | 0.470 | 2.21% | 5 |
| 2024 | 537.86% | 6.251 | 3.85% | 161 |
| 2025 | 62.44% | 3.648 | 1.55% | 78 |
| 2026YTD | 79.07% | 5.039 | 1.29% | 43 |
| recent60 | 95.27% | 5.753 | 1.09% | 20 |

### 贡献剥离

贡献诊断：

- 最大贡献股票：`300254.SZ`
- 最大贡献月份：`202406`
- 最大 1% 代理贡献阈值：`0.06668800376670787`

掘金剥离复跑：

| 剥离场景 | 年化 | Sharpe | 最大回撤 | 判断 |
|---|---:|---:|---:|---|
| 去掉最大贡献股票 | 434.73% | 4.407 | 3.83% | 年化跌破 500 |
| 去掉最大贡献月份 | 377.04% | 4.330 | 3.83% | 年化跌破 500 |
| 去掉最大 1% 交易 | 379.08% | 4.333 | 3.83% | 年化跌破 500 |
| 去掉最大股票和最大月份 | 325.60% | 4.288 | 3.83% | 年化明显跌破 500 |

### 当前准入判断

`fw_soft_all_weak85_strong105` 是当前硬指标最优版本：

- 年化达到 `500%+`；
- Sharpe 达到 `4+`；
- 最大回撤远低于 `40%`；
- 规则使用可观测特征，不使用日期硬编码；
- 回测为掘金口径。

但它仍未通过完整强准入：

- 2024 贡献仍显著高于 2025/2026；
- 去掉最大贡献股票、最大贡献月份或最大 1% 交易后，年化均跌破 500；
- 因此仍存在头部贡献依赖，不能称为平滑、低偶然、可直接投产版本。

当前状态更新为：

**当前硬指标最优研究候选：`fw_soft_all_weak85_strong105`。**

**准入状态：硬指标通过；完整强准入未通过；不建议发布生产。**

### 新增证据路径

- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_cap91_feature_weight_grid_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_cap91_feature_weight_grid_signals/fw_soft_all_weak85_strong105.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_fw_soft_slice_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_fw_soft_fragility_results.csv`
- `D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/fw_soft_fragility_diagnostics.json`
