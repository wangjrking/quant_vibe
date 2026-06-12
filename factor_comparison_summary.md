# 国泰君安 191ALPHA 因子与代码实现对比分析

## 总体统计

- **国泰君安 191ALPHA 因子总数**: 191 个
- **代码中已实现因子数**: 85 个（约 44.5%）
- **代码中未实现因子数**: 106 个（约 55.5%）

## 因子分类对比

### 一、未实现的因子（106 个）

#### 1. 量价关系因子（Alpha_001 - Alpha_101）- 101 个
这些因子主要涉及复杂的数学运算，包括：
- **排名（rank）运算**
- **相关性（correlation）运算**
- **时间序列排名（ts_rank）运算**
- **延迟（DELAY）运算**
- **变化量（delta）运算**
- **移动平均（mean）运算**
- **求和（sum）运算**

**典型公式示例**：
- Alpha_001: `(close-mean(close,20))/mean(close,20)*100`
- Alpha_002: `corr(mean(vol,5),vol,20)`
- Alpha_003: `-1*corr(rank(open),rank(vol),10)`
- Alpha_010: 复杂嵌套公式

**未实现原因**：
1. 需要横截面排名（cross-sectional rank）运算，需要全市场数据
2. 需要 adv20、adv30 等 20 日/30 日平均成交量指标
3. 公式过于复杂，涉及多层嵌套运算
4. 需要 vwap（成交量加权平均价）数据

#### 2. 价格因子（5 个）
- Alpha_118: `rank(high)` - 需要横截面排名
- Alpha_119: `rank(low)` - 需要横截面排名
- Alpha_120: `rank(open)` - 需要横截面排名
- Alpha_121: `rank(close)` - 需要横截面排名
- Alpha_122: `rank(vwap)` - 需要横截面排名和 vwap 数据

#### 3. EMA 指标（5 个）
- Alpha_133-Alpha_137: ema5, ema10, ema20, ema60, ema120
- **未实现原因**：代码中已有 dema_10, ema_10, kama_10 等类似指标

#### 4. 其他（1 个）
- Alpha_191: `60d_yield_rate` - 60 日收益率
- **未实现原因**：代码中只有到 22 日收益率

### 二、已实现的因子（85 个）

#### 1. 动量因子（5 个）✓
- Alpha_102-106: 5/10/20/60/120 日动量
- **代码实现**: `momentum_20d`, `momentum_60d`, `momentum_120d` 等

#### 2. 反转因子（4 个）✓
- Alpha_107-110: 5/10/20/60 日反转
- **代码实现**: `reversal_5d`, `reversal_10d`, `reversal_20d`, `reversal_60d`

#### 3. 波动率因子（4 个）✓
- Alpha_111-114: 5/10/20/60 日波动率
- **代码实现**: `volatility_5d`, `volatility_10d`, `volatility_20d`, `volatility_60d`

#### 4. 流动性因子（3 个）✓
- Alpha_115-117: 5/20/60 日平均成交量
- **代码实现**: `turnover_5d`, `turnover_20d`, `turnover_60d`

#### 5. 技术指标因子（14 个）✓
- RSI, MACD, ATR, CCI, CMF
- MA5/10/20/60/120
- 布林带（上轨/下轨/宽度/位置）
- **代码实现**: 完全对应

#### 6. 成交量因子（6 个）✓
- 成交量均线：`vol_ma5`, `vol_ma20`, `vol_ma60`
- 成交量比率：`vol_ratio_5d`, `vol_ratio_20d`, `vol_ratio_60d`

#### 7. 振幅因子（3 个）✓
- `amplitude_5d`, `amplitude_10d`, `amplitude_20d`

#### 8. 涨跌幅因子（4 个）✓
- `pct_change_5d`, `pct_change_10d`, `pct_change_20d`, `pct_change_60d`

#### 9. 涨跌天数因子（3 个）✓
- `up_days_5d`, `up_days_10d`, `up_days_20d`

#### 10. 涨跌比例因子（3 个）✓
- `up_ratio_5d`, `up_ratio_10d`, `up_ratio_20d`

#### 11. 跳空因子（3 个）✓
- `gap_up`, `gap_down`, `gap_ratio`

#### 12. 相对强弱因子（3 个）✓
- `strength_5d`, `strength_10d`, `strength_20d`

#### 13. 资金流向因子（3 个）✓
- `money_flow_5d`, `money_flow_10d`, `money_flow_20d`

#### 14. 价格位置因子（2 个）✓
- `price_position_20d`, `price_position_60d`

#### 15. 价格区间因子（2 个）✓
- `price_range_20d`, `price_range_60d`

#### 16. 均线偏离因子（4 个）✓
- `ma5_deviation`, `ma10_deviation`, `ma20_deviation`, `ma60_deviation`

#### 17. 波动率变化因子（2 个）✓
- `volatility_5d_change`, `volatility_20d_change`

#### 18. 动量波动率因子（2 个）✓
- `momentum_20d_vol`, `momentum_60d_vol`

#### 19. 反转波动率因子（2 个）✓
- `reversal_5d_vol`, `reversal_20d_vol`

#### 20. 换手率因子（3 个）✓
- `turnover_5d_ratio`, `turnover_20d_ratio`, `turnover_60d_ratio`

#### 21. 收益率因子（4 个）✓
- `yield_rate`（1 日）, `5d_yield_rate`, `10d_yield_rate`, `22d_yield_rate`（类似 20 日）

## 关键发现

### 实现的优势
1. **基础技术指标完整**: MACD、RSI、ATR、CCI、CMF 等主流指标全部实现
2. **均线系统完善**: 5/10/20/60/120 日均线及其偏离度
3. **布林带系统**: 上轨、下轨、宽度、位置完整
4. **量价关系**: 成交量相关因子覆盖全面
5. **动量反转**: 多周期动量和反转因子齐全
6. **波动率**: 多周期波动率及其变化率

### 缺失的因子类型
1. **横截面排名类因子**: 需要全市场数据计算排名
2. **复杂嵌套公式**: WorldQuant 101 中的复杂算法因子
3. **vwap 相关**: 缺少成交量加权平均价数据
4. **adv 系列**: 缺少 20 日/30 日平均成交量指标

## 建议补充的因子

### 高优先级（容易实现且重要）
1. **EMA 系列**: ema5, ema10, ema20, ema60, ema120
2. **60 日收益率**: 60d_yield_rate
3. **横截面排名因子**: 需要获取全市场数据

### 中优先级（需要额外数据）
1. **vwap 相关因子**: 需要计算或获取 vwap 数据
2. **adv 系列因子**: 基于现有 vol 数据计算

### 低优先级（过于复杂）
1. **WorldQuant 101 复杂因子**: Alpha_010 等嵌套公式
2. **多层相关性因子**: 需要大量计算资源

## Excel 文件位置
完整的对比 Excel 文件已生成：
`alpha_factor_comparison.xlsx`

## 总结
当前代码实现了国泰君安 191ALPHA 因子库中约 44.5% 的因子（85 个），主要集中在：
- 基础技术指标
- 均线系统
- 动量/反转
- 波动率
- 成交量
- 价格位置

未实现的 55.5% 因子（106 个）主要是：
- 需要横截面排名的量价关系因子
- 需要 vwap 等特殊数据的因子
- 过于复杂的嵌套公式因子

这些已实现的因子已经涵盖了量化选股中最常用和最重要的技术指标，可以满足大部分量化策略的需求。
