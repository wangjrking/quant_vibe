# 流动性精选一只 v2 可复现合约

## 当前结论

旧投产策略无法通过“现有数据重训”精确复现，但可以通过三张已归档组件预测表精确重建最终投产预测表。

重建验证结果：

- 目标表：`stock_predict_data_executable_5d_open_return_prod_aligned_rebuilt_20260614`
- 参考表：`stock_predict_data_executable_5d_open_return_prod_aligned_202206_202606`
- 行数：`1,253,476`
- 交易日：`976`
- 日期范围：`20220606-20260612`
- 缺失 key：`0`
- 多出 key：`0`
- `pred_prob` 差异：`0`

## 三段来源

| 区间 | 来源表 | 行数 |
|---|---|---:|
| 20220606-20240603 | `stock_predict_data_executable_5d_open_return_prod_rebuild_aligned_early_202206_202406` | 370682 |
| 20240604-20260609 | `stock_predict_data_executable_5d_open_return_m018_scores_currentdata_202406_202606` | 867177 |
| 20260610-20260612 | `stock_predict_data_executable_5d_open_return_prod_aligned_increment_20260610_20260612` | 15617 |

## 禁止事项

- 禁止用当前全 A 数据重训结果冒充旧投产版本。
- 禁止只保存参数不保存模型 checkpoint。
- 禁止只保存静态股票池，不保存每日 universe 或样本 hash。

## 后续新模型必须保存

- 原始数据快照版本或数据库 hash。
- 每日 universe 文件。
- 每折训练样本 key 文件和 hash。
- 每折特征列表和 hash。
- 模型参数 JSON。
- XGBoost booster/model 文件。
- 预测表 hash。
- 信号表 hash。
- 掘金回测日志和指标 JSON。
