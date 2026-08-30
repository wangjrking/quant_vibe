# fw_soft 发布生产阻断说明

## 结论

用户已明确要求将 `fw_soft_all_weak85_strong105` 覆盖发布到生产，但发布前自审发现硬阻断：

1. 研究版 `fw_soft_all_weak85_strong105` 信号只覆盖到：
   - `signal_date_max=20260528`
   - `buy_date_max=20260529`
2. 当前生产策略 `prod_deepdrop_smallcap_refill_v20260704` 的 full-history 已覆盖到：
   - `signal_date_max=20260703`
   - `buy_date_max=20260706`
3. 若直接把研究版发布为生产，会导致生产 latest signal 倒退，且不能满足当前 L5/L6 正式链路的 latest signal 要求。
4. 已按同一仓位规则从当前生产 full-history 重建一版 `fw_soft`，并用掘金复跑；复跑结果不满足研究版 501% 指标。

因此本次没有更新 `registry.json`，没有覆盖当前生产策略。

## 原研究版指标

研究版信号：

`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/juejin_cap91_feature_weight_grid_signals/fw_soft_all_weak85_strong105.csv`

掘金结果：

| 指标 | 数值 |
|---|---:|
| 年化 | 501.69% |
| Sharpe | 4.442 |
| 最大回撤 | 3.83% |
| 开仓/平仓 | 287 / 287 |

但该信号日期不完整：

| 字段 | 数值 |
|---|---|
| `signal_date_min` | `20230817` |
| `signal_date_max` | `20260528` |
| `buy_date_min` | `20230818` |
| `buy_date_max` | `20260529` |

## 当前生产 full-history 重建验证

重建脚本：

`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/rebuild_fw_soft_from_current_production_signal.py`

输入：

`D:/work/quant/quant_mcp/quant/main/strategy_library/production/prod_deepdrop_smallcap_refill_v20260704/signals/full_history_deepdrop_smallcap_refill.csv`

输出：

`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/production_publish_candidates/fw_soft_all_weak85_strong105_from_current_production_full_history.csv`

重建后覆盖：

| 字段 | 数值 |
|---|---|
| `row_count` | `314` |
| `signal_date_max` | `20260703` |
| `buy_date_max` | `20260706` |
| `duplicate_buy_stock_keys` | `0` |
| `latest_buy_date_rows` | `3` |
| `latest_buy_date_target_pct_sum` | `0.91` |

掘金复跑日志：

`D:/work/quant/quant_mcp/quant/data_file/reports/strategy_agent_active_l4_param_search_20260705/production_publish_candidates/fw_soft_all_weak85_strong105_from_current_production_full_history_slip0p0030.log`

掘金复跑结果：

| 指标 | 数值 |
|---|---:|
| 年化 | 257.06% |
| Sharpe | 3.646 |
| 最大回撤 | 3.56% |
| 开仓/平仓 | 311 / 311 |
| 胜率 | 70.42% |

该结果未达到用户要求的年化 `500%` 和 Sharpe `4`。

## 规则差异解释

研究版 `fw_soft_all_weak85_strong105` 是基于旧研究信号集合 `blend3_scale1p1_cap91` 做仓位微调；该信号集合只到 `20260529`。

当前生产 full-history 是 `prod_deepdrop_smallcap_refill_v20260704` 的生产信号集合，覆盖到 `20260706`。将同一仓位微调规则应用到当前 full-history 后，信号集合发生变化，因此掘金指标不再等于研究版 501%。

## 发布判断

本次发布请求的安全结论：

- 不应把研究版 501% 指标写入生产 registry；
- 不应把只到 `20260529` 的研究信号覆盖生产 latest signal；
- 不应把当前 full-history 重建版作为 501% 策略发布，因为掘金复跑只得到 `257.06% / 3.646`；
- 当前生产仍保持 `prod_deepdrop_smallcap_refill_v20260704`。

## 下一步建议

如仍要发布，有两个可选路径：

1. 接受降级指标：将当前 full-history 重建版作为新生产策略发布，指标按 `257.06% / Sharpe 3.646 / MDD 3.56%` 记录。
2. 继续研究：重新用当前 active formal L4 资产构建完整到最新日期的 `blend3/cap91/fw_soft` 同口径信号，再以掘金复跑结果决定是否发布。
