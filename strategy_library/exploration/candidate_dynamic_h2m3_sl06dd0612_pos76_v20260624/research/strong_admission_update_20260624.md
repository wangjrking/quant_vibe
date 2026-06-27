# 强准入更新记录

## 当前口径

收益类指标只作为弱准入，不能单独决定生产准入。敏感性、持续性、低路径依赖、硬过滤审计、未来信息排除、掘金复现、生产归档契约属于强准入。

## 回撤缩放邻域

固定 `pos76 + sl06 + dynamic_h2_m3`，只调整账户回撤缩放：

| 版本 | 年化 | Sharpe | 最大回撤 | recent60 年化 | 近期开仓最差年化 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pos76_sl06_no_dd` | 476.92% | 1.92 | 49.84% | 78.59% | 163.26% | 收益最高，但回撤显著放大 |
| `pos76_sl06_dd0612_s8060` | 395.64% | 2.02 | 37.01% | 44.79% | 161.88% | 当前主线，收益和回撤更均衡 |
| `pos76_sl06_dd0814_s8565` | 392.25% | 1.98 | 40.42% | 36.72% | 161.32% | 年化接近主线，但回撤更高、近期更弱 |
| `pos76_sl06_dd0510_s8060` | 326.56% | 1.94 | 37.45% | 47.19% | 162.28% | 回撤接近主线，但收益明显降低 |

判断：`no_dd` 不能仅因年化更高替代主线。其敏感性和近期开仓不差，但最大回撤接近 50%，不符合当前对生产候选的风险约束倾向。

## no_dd 降仓位邻域

关闭账户回撤缩放后，只调整仓位：

| 版本 | 年化 | Sharpe | 最大回撤 | recent60 年化 | 近期开仓最差年化 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `no_dd_pos76_sl06` | 476.92% | 1.92 | 49.84% | 78.59% | 163.26% | 收益最高，回撤过大 |
| `no_dd_pos70_sl06` | 396.99% | 1.97 | 46.87% | 75.25% | 151.28% | 年化接近主线，但回撤仍高 |
| `no_dd_pos66_sl06` | 350.55% | 2.00 | 44.83% | 73.88% | 134.67% | 近期较好，但收益和近期开仓弱于主线 |
| `no_dd_pos58_sl06` | 267.42% | 2.05 | 40.52% | 18.07% | 118.04% | 回撤下降，但近期持续性变差 |

判断：降低仓位可以压低回撤，但没有出现“收益、回撤、recent60、近期开仓”同时优于当前 `dd0612` 主线的点。

## 当前结论

当前仍保留 `dynamic_h2_m3_c098_w78_5d12_3d10_pos76_e097_mh1_sl06_dd0612` 作为更接近生产准入的主线候选：

- 年化：395.64%
- Sharpe：2.02
- 最大回撤：37.01%
- recent60 年化：44.79%
- 近期开仓最差年化：161.88%
- 硬过滤审计：北交所、ST/风险警示、退市、开盘涨停买入均为 0

但该候选仍未正式发布生产，仍需审计复核和用户明确授权。

## 证据路径

- 回撤缩放邻域：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_pos76_dd_neighborhood_20260624`
- no_dd 仓位邻域：`D:\work\quant\quant_mcp\quant\data_file\reports\strategy_agent_goal_high_annual_20260623\top1_no_delist_rerun_20260624\diversification_tune_20260624\strict_sync_liquidity_neighborhood_20260624\formal_horizon_entry_confirmation_20260624\dynamic_h2_m3_no_dd_position_20260624`
- 候选归档：`D:\work\quant\quant_mcp\quant\main\strategy_library\exploration\candidate_dynamic_h2m3_sl06dd0612_pos76_v20260624`
