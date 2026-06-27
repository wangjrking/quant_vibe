# 卖出信号优化结果说明

## 结论

本轮只围绕当前 L5 研究候选 `best_noncal_avgpred205` 的卖出和续持参数做小扰动，未引入行业限制、月份排除、日期排除或 1D 模型结果。结果显示：单纯通过放宽卖出、延后分数卖出、关闭分数卖出、改变每日最大卖出数，可以提高部分时段的持仓暴露，但不能在不降低收益质量的前提下稳定提升持仓。

当前仍建议保留主候选 `best_noncal_avgpred205`；若目标偏向“年化不降、持仓略升”，可以把 `best_noncal_avgpred205_rw2121201919` 作为备选送审，但它的 Sharpe 略低于主候选。

## 对比结果

| 候选 | 年化收益率 | Sharpe | 最大回撤 | 平均持仓率 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `best_noncal_avgpred205` | 254.11% | 3.0066 | 23.60% | 58.89% | 当前主候选 |
| `best_noncal_avgpred205_rw2121201919` | 257.09% | 3.0045 | 23.81% | 59.16% | 轻微提高前两名权重，年化和持仓略升，Sharpe 略降但仍过 3 |
| `best_noncal_avgpred205_exit099` | 177.68% | 2.6873 | 29.45% | 60.60% | 放宽分数卖出阈值，持仓升但收益和 Sharpe 明显下降 |
| `best_noncal_avgpred205_minh4` | 173.60% | 2.3598 | 31.60% | 59.33% | 分数卖出最小持有期从 3 天延到 4 天，收益质量下降 |
| `best_noncal_avgpred205_sell0` | 139.31% | 2.2313 | 25.45% | 52.30% | 禁止日度分数卖出，持仓反而下降，收益下降 |
| `best_noncal_avgpred205_sell2` | 129.19% | 2.0997 | 25.46% | 52.57% | 每日最多卖出 2 只，收益下降 |
| `best_noncal_avgpred205_no_score_exit` | 110.41% | 1.8112 | 26.83% | 59.48% | 关闭日度分数卖出，持仓略升但收益质量明显下降 |

## 判断

当前收益主要来自高质量信号日和较快剔除弱续持标的。卖出规则不是简单的“卖得太快导致仓位低”，而是在过滤低质量延续持仓。把卖出放宽后，确实可能增加局部持仓或 `ge80_ratio`，但会引入更多弱持仓，导致年化、Sharpe 或回撤恶化。

在“年化不降低、Sharpe 不跌破 3”的约束下，本轮没有找到纯卖出信号优化带来的有效提升。最接近可用的改进是 `best_noncal_avgpred205_rw2121201919`，但它属于仓位权重微调，不属于卖出信号优化。

## 证据路径

- 掘金回测汇总：`D:/work/quant/quant_mcp/quant/main/strategy_library/exploration/l5_formal_5d10d_candidate_20260621/research/summary_noncalendar_quality_refine.csv`
- 年化排序：`D:/work/quant/quant_mcp/quant/main/strategy_library/exploration/l5_formal_5d10d_candidate_20260621/research/summary_noncalendar_quality_refine_by_annual.csv`
- Sharpe 排序：`D:/work/quant/quant_mcp/quant/main/strategy_library/exploration/l5_formal_5d10d_candidate_20260621/research/summary_noncalendar_quality_refine_by_sharpe.csv`
- 原始运行目录：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_model_application_20260621/formal_5d10d_noncalendar_quality_refine_20260621`
- 研究脚本：`D:/work/quant/quant_mcp/quant/main/research_formal_5d10d_noncalendar_quality_refine_20260621.py`
