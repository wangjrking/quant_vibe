# 近四年 L4 dd_loose 与多票分散阶段报告

## 当前可靠结论

本轮在最新 L4 近四年 formal 资产上继续调参后，得到两个结论：

1. `Top1 / 60% / dd_loose` 是当前更接近强准入的候选。
2. `Top2 / Top3` 分散持仓显著提高全周期收益、夏普和回撤表现，但低路径依赖不通过，暂不能作为生产准入候选。

## 当前更接近准入的候选

候选：

`w72_23_05_amt150_mv30__hold3m5_c097_e096_ddtight_pos65__strict_amt150000_mv300000_pc150_pos60`

新增风控参数：

- 账户回撤软触发：8%
- 账户回撤硬触发：14%
- 恢复触发：4%
- 软缩放：0.80
- 硬缩放：0.60

该版本记为 `dd_loose`。

## dd_loose 掘金结果

| 口径 | 年化 | 累计收益 | 夏普 | 最大回撤 | recent120 | recent60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 买入端自适应滑点、卖出低摩擦 | 114.14% | 463.13% | 0.786 | 35.85% | 10.53% | 9.49% |
| 买入端自适应滑点、卖出 0.25 倍压力 | 109.73% | 445.23% | 0.770 | 36.43% | 4.07% | 8.27% |

相对上一版 60% 基准缩仓，dd_loose 的改善点：

- 全周期收益明显提高。
- 卖出 0.25 倍压力下 recent120 / recent60 均保持正收益。
- 低路径依赖所有检查启动点均为正。

代价：

- 最大回撤从约 28.44% 上升到约 35.85%。
- 夏普低于基准缩仓版本。
- 2025-07 附近冷启动最低年化仍只有 0.35%，底线偏弱。

## dd_loose 低路径依赖

| 锚点 | 启动点数 | 年化最小值 | 年化中位数 | 年化最大值 | 最低夏普 | 最大回撤上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-01-02 | 7 | 9.73% | 10.67% | 11.15% | 0.472 | 33.36% |
| 2025-01-02 | 7 | 34.57% | 38.72% | 40.67% | 1.206 | 23.53% |
| 2025-07-01 | 7 | 0.35% | 2.28% | 3.63% | 0.121 | 21.15% |
| 2025-10-09 | 7 | 7.14% | 9.73% | 19.12% | 0.414 | 15.28% |
| 2026-01-05 | 7 | 19.32% | 25.35% | 37.34% | 0.736 | 17.12% |

结论：低路径依赖形式上通过“所有启动点为正”，但 2025-07 附近收益底线仍很薄，不能称为强稳健。

## 多票分散结果

| 策略 | 卖出压力 | 年化 | 夏普 | 最大回撤 | recent120 | recent60 | 低路径依赖 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Top3 / 每票 25% | 0.25 | 341.82% | 1.060 | 27.65% | 3.25% | 12.80% | 不通过 |
| Top3 / 每票 20% | 0.25 | 195.39% | 1.158 | 23.53% | 3.20% | 16.42% | 不通过 |
| Top2 / 每票 30% | 0.25 | 160.34% | 0.919 | 34.92% | 21.01% | 30.56% | 不通过 |
| Top2 / 每票 35% | 0.25 | 228.27% | 0.873 | 37.81% | 16.58% | 37.00% | 未继续复核 |

多票分散是有效研究方向，但目前不满足低路径依赖强准入。

## 硬过滤审计

`Top1 / 60% / dd_loose` 信号审计通过：

- 信号行数：985
- 信号日：985
- 买入日：984
- 北交所命中：0
- ST/风险警示命中：0
- 退市命中：0
- 买入日涨停命中：0
- 底表缺失：0

多票分散候选的硬过滤审计也通过：

- `Top3 / 25%`：2955 行，硬过滤命中 0。
- `Top3 / 20%`：2955 行，硬过滤命中 0。
- `Top2 / 30%`：1970 行，硬过滤命中 0。

## 贡献集中度说明

已尝试从掘金日志解析交易贡献，但解析交易数与掘金 `close_count` 不一致，且简化解析总 PnL 与掘金累计收益无法对齐。因此当前 `style_concentration_audit.json` 只能作为初审线索，不能作为正式贡献集中度强证据。

正式准入前仍需补一版与掘金成交记录一致的贡献集中度审计。

## 当前准入判断

- `Top1 / 60% / dd_loose`：研究候选通过，当前最接近准入，但仍需补贡献集中度正式审计和风格暴露/漂移审计。
- `Top2 / Top3`：收益表现更好，但低路径依赖失败，暂不准入。
- 当前尚不能宣称“生产准入完全通过”。

## 证据路径

- dd_loose 指标：`quant/data_file/reports/strategy_agent_adaptive70w_fouryear_pos60_ddrisk_20260629/ddrisk_summary.csv`
- dd_loose 低路径：`quant/data_file/reports/strategy_agent_adaptive70w_fouryear_pos60_ddloose_admission_20260629/lowpath_summary.csv`
- dd_loose 硬审计：`quant/data_file/reports/strategy_agent_adaptive70w_fouryear_pos60_ddloose_admission_20260629/hard_gate_audit.json`
- 多票分散指标：`quant/data_file/reports/strategy_agent_adaptive70w_fouryear_multitop_ddloose_20260629/multitop_summary.csv`
- Top3/25 低路径：`quant/data_file/reports/strategy_agent_adaptive70w_fouryear_top3pos25_admission_20260629/lowpath_summary.csv`
- Top3/20 低路径：`quant/data_file/reports/strategy_agent_adaptive70w_fouryear_top3pos20_admission_20260629/lowpath_summary.csv`
- Top2/30 低路径：`quant/data_file/reports/strategy_agent_adaptive70w_fouryear_top2pos30_admission_20260629/lowpath_summary.csv`
