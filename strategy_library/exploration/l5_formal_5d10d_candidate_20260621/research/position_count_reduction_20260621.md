# 减少持股数量测试说明

## 结论

本轮在当前主候选 `best_noncal_avgpred205` 基础上，补充测试了 Top4 持股数量方案，并与既有 Top5、Top3、Top2、Top1 结果对比。掘金回测结果显示：减少选股数量不能提升当前策略效果，Top5 仍是收益、Sharpe 和回撤之间最均衡的组合。

## 对比结果

| 候选 | 最大持股数 | 年化收益率 | Sharpe | 最大回撤 | 平均持仓率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best_noncal_avgpred205` | 5 | 254.11% | 3.0066 | 23.60% | 58.89% |
| `best_noncal_avgpred205_rw2121201919` | 5 | 257.09% | 3.0045 | 23.81% | 59.16% |
| `best_noncal_avgpred205_top4_25252424` | 4 | 189.69% | 2.2374 | 30.17% | 63.03% |
| `best_noncal_avgpred205_top4_30272517` | 4 | 187.84% | 2.2855 | 31.46% | 59.43% |
| `best_noncal_avgpred205_top4_28262420` | 4 | 180.44% | 2.3020 | 29.94% | 58.73% |
| `best_noncal_avgpred205_top3_302928` | 3 | 177.17% | 2.1989 | 28.65% | 55.71% |
| `best_noncal_avgpred205_top2_4949` | 2 | 129.52% | 1.5459 | 34.71% | 59.17% |
| `best_noncal_avgpred205_top1_098_exit099` | 1 | 238.74% | 1.7733 | 33.25% | 65.16% |

## 判断

当前模型分数的有效信息不只集中在第 1 名或前 3 名。减少持股数量后，组合分散度下降，个股波动暴露变大，Sharpe 明显下降。Top4 虽然部分版本能提高平均持仓率，但年化收益和 Sharpe 都低于 Top5 主候选。

因此，当前不建议为了“少买几支”替换主候选。若后续目标变成降低交易复杂度，可以单独以“持股数约束”为目标重新定义优化函数；但按当前收益和 Sharpe 目标，Top5 更优。

## 证据路径

- 掘金回测汇总：`D:/work/quant/quant_mcp/quant/main/strategy_library/exploration/l5_formal_5d10d_candidate_20260621/research/summary_noncalendar_quality_refine.csv`
- 年化排序：`D:/work/quant/quant_mcp/quant/main/strategy_library/exploration/l5_formal_5d10d_candidate_20260621/research/summary_noncalendar_quality_refine_by_annual.csv`
- Sharpe 排序：`D:/work/quant/quant_mcp/quant/main/strategy_library/exploration/l5_formal_5d10d_candidate_20260621/research/summary_noncalendar_quality_refine_by_sharpe.csv`
- 原始运行目录：`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_model_application_20260621/formal_5d10d_noncalendar_quality_refine_20260621`
- 研究脚本：`D:/work/quant/quant_mcp/quant/main/research_formal_5d10d_noncalendar_quality_refine_20260621.py`
