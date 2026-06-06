# Alpha158 因子清单（data_process_module.py 新增）

本文档列出了 `data_process_module.py` 中新增的所有 Alpha158 因子。

---

## 📊 因子总数：109 个

---

## 一、K线基础因子（9个）

| 序号 | 因子名称 | 计算公式 | 含义 |
|------|----------|----------|------|
| 1 | alpha158_kmid | `(close - open) / open` | 收盘与开盘价差比例 |
| 2 | alpha158_klen | `(high - low) / open` | 最高最低价差比例 |
| 3 | alpha158_kmid2 | `(close - open) / (high - low)` | 收开差与振幅比 |
| 4 | alpha158_kup | `(high - max(open, close)) / open` | 上影线比例 |
| 5 | alpha158_kup2 | `(high - max(open, close)) / (high - low)` | 上影线占比 |
| 6 | alpha158_klow | `(min(open, close) - low) / open` | 下影线比例 |
| 7 | alpha158_klow2 | `(min(open, close) - low) / (high - low)` | 下影线占比 |
| 8 | alpha158_ksft | `(2*close - high - low) / open` | 收盘价相对中点位置 |
| 9 | alpha158_ksft2 | `(2*close - high - low) / (high - low)` | 收盘价在振幅中位置 |

---

## 二、趋势类因子（15个）- 多周期窗口

### 2.1 ROC 价格变化率（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 10 | alpha158_roc5 | `close / close.shift(5)` | 5天 |
| 11 | alpha158_roc10 | `close / close.shift(10)` | 10天 |
| 12 | alpha158_roc20 | `close / close.shift(20)` | 20天 |
| 13 | alpha158_roc30 | `close / close.shift(30)` | 30天 |
| 14 | alpha158_roc60 | `close / close.shift(60)` | 60天 |

### 2.2 MA 移动平均比（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 15 | alpha158_ma5 | `rolling(5).mean() / close` | 5天 |
| 16 | alpha158_ma10 | `rolling(10).mean() / close` | 10天 |
| 17 | alpha158_ma20 | `rolling(20).mean() / close` | 20天 |
| 18 | alpha158_ma30 | `rolling(30).mean() / close` | 30天 |
| 19 | alpha158_ma60 | `rolling(60).mean() / close` | 60天 |

### 2.3 STD 标准差比（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 20 | alpha158_std5 | `rolling(5).std() / close` | 5天 |
| 21 | alpha158_std10 | `rolling(10).std() / close` | 10天 |
| 22 | alpha158_std20 | `rolling(20).std() / close` | 20天 |
| 23 | alpha158_std30 | `rolling(30).std() / close` | 30天 |
| 24 | alpha158_std60 | `rolling(60).std() / close` | 60天 |

---

## 三、波动类因子（25个）- 多周期窗口

### 3.1 MAX 最高价比（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 25 | alpha158_max5 | `rolling(5).max() / close` | 5天 |
| 26 | alpha158_max10 | `rolling(10).max() / close` | 10天 |
| 27 | alpha158_max20 | `rolling(20).max() / close` | 20天 |
| 28 | alpha158_max30 | `rolling(30).max() / close` | 30天 |
| 29 | alpha158_max60 | `rolling(60).max() / close` | 60天 |

### 3.2 MIN 最低价比（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 30 | alpha158_min5 | `rolling(5).min() / close` | 5天 |
| 31 | alpha158_min10 | `rolling(10).min() / close` | 10天 |
| 32 | alpha158_min20 | `rolling(20).min() / close` | 20天 |
| 33 | alpha158_min30 | `rolling(30).min() / close` | 30天 |
| 34 | alpha158_min60 | `rolling(60).min() / close` | 60天 |

### 3.3 QTLU 80%分位比（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 35 | alpha158_qtlu5 | `rolling(5).quantile(0.8) / close` | 5天 |
| 36 | alpha158_qtlu10 | `rolling(10).quantile(0.8) / close` | 10天 |
| 37 | alpha158_qtlu20 | `rolling(20).quantile(0.8) / close` | 20天 |
| 38 | alpha158_qtlu30 | `rolling(30).quantile(0.8) / close` | 30天 |
| 39 | alpha158_qtlu60 | `rolling(60).quantile(0.8) / close` | 60天 |

### 3.4 QTLD 20%分位比（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 40 | alpha158_qtld5 | `rolling(5).quantile(0.2) / close` | 5天 |
| 41 | alpha158_qtld10 | `rolling(10).quantile(0.2) / close` | 10天 |
| 42 | alpha158_qtld20 | `rolling(20).quantile(0.2) / close` | 20天 |
| 43 | alpha158_qtld30 | `rolling(30).quantile(0.2) / close` | 30天 |
| 44 | alpha158_qtld60 | `rolling(60).quantile(0.2) / close` | 60天 |

### 3.5 RSV 相对强弱值（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 45 | alpha158_rsv5 | `(close - min) / (max - min)` | 5天 |
| 46 | alpha158_rsv10 | `(close - min) / (max - min)` | 10天 |
| 47 | alpha158_rsv20 | `(close - min) / (max - min)` | 20天 |
| 48 | alpha158_rsv30 | `(close - min) / (max - min)` | 30天 |
| 49 | alpha158_rsv60 | `(close - min) / (max - min)` | 60天 |

---

## 四、极值位置因子（15个）

### 4.1 IMAX 最高价出现位置（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 50 | alpha158_imax5 | `argmax(high, 5) / 5` | 5天 |
| 51 | alpha158_imax10 | `argmax(high, 10) / 10` | 10天 |
| 52 | alpha158_imax20 | `argmax(high, 20) / 20` | 20天 |
| 53 | alpha158_imax30 | `argmax(high, 30) / 30` | 30天 |
| 54 | alpha158_imax60 | `argmax(high, 60) / 60` | 60天 |

### 4.2 IMIN 最低价出现位置（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 55 | alpha158_imin5 | `argmin(low, 5) / 5` | 5天 |
| 56 | alpha158_imin10 | `argmin(low, 10) / 10` | 10天 |
| 57 | alpha158_imin20 | `argmin(low, 20) / 20` | 20天 |
| 58 | alpha158_imin30 | `argmin(low, 30) / 30` | 30天 |
| 59 | alpha158_imin60 | `argmin(low, 60) / 60` | 60天 |

### 4.3 IMXD 高低点跨度（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 60 | alpha158_imxd5 | `imax - imin` | 5天 |
| 61 | alpha158_imxd10 | `imax - imin` | 10天 |
| 62 | alpha158_imxd20 | `imax - imin` | 20天 |
| 63 | alpha158_imxd30 | `imax - imin` | 30天 |
| 64 | alpha158_imxd60 | `imax - imin` | 60天 |

---

## 五、价量统计类因子（25个）

### 5.1 CORR 价格成交量相关性（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 65 | alpha158_corr5 | `corr(close, log(vol), 5)` | 5天 |
| 66 | alpha158_corr10 | `corr(close, log(vol), 10)` | 10天 |
| 67 | alpha158_corr20 | `corr(close, log(vol), 20)` | 20天 |
| 68 | alpha158_corr30 | `corr(close, log(vol), 30)` | 30天 |
| 69 | alpha158_corr60 | `corr(close, log(vol), 60)` | 60天 |

### 5.2 CORD 变化相关性（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 70 | alpha158_cord5 | `corr(pct_close, pct_vol, 5)` | 5天 |
| 71 | alpha158_cord10 | `corr(pct_close, pct_vol, 10)` | 10天 |
| 72 | alpha158_cord20 | `corr(pct_close, pct_vol, 20)` | 20天 |
| 73 | alpha158_cord30 | `corr(pct_close, pct_vol, 30)` | 30天 |
| 74 | alpha158_cord60 | `corr(pct_close, pct_vol, 60)` | 60天 |

### 5.3 CNTP 上涨天数比例（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 75 | alpha158_cntp5 | `mean(close > ref(close,1), 5)` | 5天 |
| 76 | alpha158_cntp10 | `mean(close > ref(close,1), 10)` | 10天 |
| 77 | alpha158_cntp20 | `mean(close > ref(close,1), 20)` | 20天 |
| 78 | alpha158_cntp30 | `mean(close > ref(close,1), 30)` | 30天 |
| 79 | alpha158_cntp60 | `mean(close > ref(close,1), 60)` | 60天 |

### 5.4 CNTN 下跌天数比例（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 80 | alpha158_cntn5 | `mean(close < ref(close,1), 5)` | 5天 |
| 81 | alpha158_cntn10 | `mean(close < ref(close,1), 10)` | 10天 |
| 82 | alpha158_cntn20 | `mean(close < ref(close,1), 20)` | 20天 |
| 83 | alpha158_cntn30 | `mean(close < ref(close,1), 30)` | 30天 |
| 84 | alpha158_cntn60 | `mean(close < ref(close,1), 60)` | 60天 |

### 5.5 CNTD 净涨天数（5个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 85 | alpha158_cntd5 | `cntp - cntn` | 5天 |
| 86 | alpha158_cntd10 | `cntp - cntn` | 10天 |
| 87 | alpha158_cntd20 | `cntp - cntn` | 20天 |
| 88 | alpha158_cntd30 | `cntp - cntn` | 30天 |
| 89 | alpha158_cntd60 | `cntp - cntn` | 60天 |

---

## 六、RSI类因子（6个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 90 | alpha158_sump6 | `sum(pos_gain) / sum(abs(change))` | 6天 |
| 91 | alpha158_sump12 | `sum(pos_gain) / sum(abs(change))` | 12天 |
| 92 | alpha158_sump24 | `sum(pos_gain) / sum(abs(change))` | 24天 |
| 93 | alpha158_sumd6 | `sum(neg_loss) / sum(abs(change))` | 6天 |
| 94 | alpha158_sumd12 | `sum(neg_loss) / sum(abs(change))` | 12天 |
| 95 | alpha158_sumd24 | `sum(neg_loss) / sum(abs(change))` | 24天 |

---

## 七、复合技术指标（11个）

### 7.1 BOLL 布林带位置

| 序号 | 因子名称 | 计算公式 |
|------|----------|----------|
| 96 | alpha158_boll | `(close - ma20) / (2 * std20)` |

### 7.2 ADX 平均趋向指数

| 序号 | 因子名称 | 计算公式 |
|------|----------|----------|
| 97 | alpha158_adx | `mean(dx, 14)` |

### 7.3 KDJ 指标（9个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 98 | alpha158_k6 | `ewm(rsv, span=3)` | 6天 |
| 99 | alpha158_k12 | `ewm(rsv, span=3)` | 12天 |
| 100 | alpha158_k24 | `ewm(rsv, span=3)` | 24天 |
| 101 | alpha158_d6 | `ewm(K, span=3)` | 6天 |
| 102 | alpha158_d12 | `ewm(K, span=3)` | 12天 |
| 103 | alpha158_d24 | `ewm(K, span=3)` | 24天 |
| 104 | alpha158_j6 | `3*K - 2*D` | 6天 |
| 105 | alpha158_j12 | `3*K - 2*D` | 12天 |
| 106 | alpha158_j24 | `3*K - 2*D` | 24天 |

---

## 八、其他因子（3个）

| 序号 | 因子名称 | 计算公式 | 窗口 |
|------|----------|----------|------|
| 107 | alpha158_rank5 | `rolling(5).rank(pct=True)` | 5天 |
| 108 | alpha158_rank10 | `rolling(10).rank(pct=True)` | 10天 |
| 109 | alpha158_rank20 | `rolling(20).rank(pct=True)` | 20天 |

---

## 📋 因子清单（纯文本版）

```
# 一、K线基础因子（9个）
alpha158_kmid
alpha158_klen
alpha158_kmid2
alpha158_kup
alpha158_kup2
alpha158_klow
alpha158_klow2
alpha158_ksft
alpha158_ksft2

# 二、趋势类因子（15个）
alpha158_roc5
alpha158_roc10
alpha158_roc20
alpha158_roc30
alpha158_roc60
alpha158_ma5
alpha158_ma10
alpha158_ma20
alpha158_ma30
alpha158_ma60
alpha158_std5
alpha158_std10
alpha158_std20
alpha158_std30
alpha158_std60

# 三、波动类因子（25个）
alpha158_max5
alpha158_max10
alpha158_max20
alpha158_max30
alpha158_max60
alpha158_min5
alpha158_min10
alpha158_min20
alpha158_min30
alpha158_min60
alpha158_qtlu5
alpha158_qtlu10
alpha158_qtlu20
alpha158_qtlu30
alpha158_qtlu60
alpha158_qtld5
alpha158_qtld10
alpha158_qtld20
alpha158_qtld30
alpha158_qtld60
alpha158_rsv5
alpha158_rsv10
alpha158_rsv20
alpha158_rsv30
alpha158_rsv60

# 四、极值位置因子（15个）
alpha158_imax5
alpha158_imax10
alpha158_imax20
alpha158_imax30
alpha158_imax60
alpha158_imin5
alpha158_imin10
alpha158_imin20
alpha158_imin30
alpha158_imin60
alpha158_imxd5
alpha158_imxd10
alpha158_imxd20
alpha158_imxd30
alpha158_imxd60

# 五、价量统计类因子（25个）
alpha158_corr5
alpha158_corr10
alpha158_corr20
alpha158_corr30
alpha158_corr60
alpha158_cord5
alpha158_cord10
alpha158_cord20
alpha158_cord30
alpha158_cord60
alpha158_cntp5
alpha158_cntp10
alpha158_cntp20
alpha158_cntp30
alpha158_cntp60
alpha158_cntn5
alpha158_cntn10
alpha158_cntn20
alpha158_cntn30
alpha158_cntn60
alpha158_cntd5
alpha158_cntd10
alpha158_cntd20
alpha158_cntd30
alpha158_cntd60

# 六、RSI类因子（6个）
alpha158_sump6
alpha158_sump12
alpha158_sump24
alpha158_sumd6
alpha158_sumd12
alpha158_sumd24

# 七、复合技术指标（11个）
alpha158_boll
alpha158_adx
alpha158_k6
alpha158_k12
alpha158_k24
alpha158_d6
alpha158_d12
alpha158_d24
alpha158_j6
alpha158_j12
alpha158_j24

# 八、其他因子（3个）
alpha158_rank5
alpha158_rank10
alpha158_rank20
```

---

## 📊 统计汇总

| 类别 | 因子数量 | 占比 |
|------|----------|------|
| K线基础因子 | 9 | 8.3% |
| 趋势类因子 | 15 | 13.8% |
| 波动类因子 | 25 | 22.9% |
| 极值位置因子 | 15 | 13.8% |
| 价量统计类因子 | 25 | 22.9% |
| RSI类因子 | 6 | 5.5% |
| 复合技术指标 | 11 | 10.1% |
| 其他因子 | 3 | 2.8% |
| **总计** | **109** | **100%** |

---

## 🐍 Python 代码引用

```python
# 获取所有新增的 alpha158 因子名称
alpha158_factors = [
    # K线基础因子（9个）
    'alpha158_kmid', 'alpha158_klen', 'alpha158_kmid2', 'alpha158_kup',
    'alpha158_kup2', 'alpha158_klow', 'alpha158_klow2', 'alpha158_ksft', 'alpha158_ksft2',
    
    # 趋势类因子（15个）
    'alpha158_roc5', 'alpha158_roc10', 'alpha158_roc20', 'alpha158_roc30', 'alpha158_roc60',
    'alpha158_ma5', 'alpha158_ma10', 'alpha158_ma20', 'alpha158_ma30', 'alpha158_ma60',
    'alpha158_std5', 'alpha158_std10', 'alpha158_std20', 'alpha158_std30', 'alpha158_std60',
    
    # 波动类因子（25个）
    'alpha158_max5', 'alpha158_max10', 'alpha158_max20', 'alpha158_max30', 'alpha158_max60',
    'alpha158_min5', 'alpha158_min10', 'alpha158_min20', 'alpha158_min30', 'alpha158_min60',
    'alpha158_qtlu5', 'alpha158_qtlu10', 'alpha158_qtlu20', 'alpha158_qtlu30', 'alpha158_qtlu60',
    'alpha158_qtld5', 'alpha158_qtld10', 'alpha158_qtld20', 'alpha158_qtld30', 'alpha158_qtld60',
    'alpha158_rsv5', 'alpha158_rsv10', 'alpha158_rsv20', 'alpha158_rsv30', 'alpha158_rsv60',
    
    # 极值位置因子（15个）
    'alpha158_imax5', 'alpha158_imax10', 'alpha158_imax20', 'alpha158_imax30', 'alpha158_imax60',
    'alpha158_imin5', 'alpha158_imin10', 'alpha158_imin20', 'alpha158_imin30', 'alpha158_imin60',
    'alpha158_imxd5', 'alpha158_imxd10', 'alpha158_imxd20', 'alpha158_imxd30', 'alpha158_imxd60',
    
    # 价量统计类因子（25个）
    'alpha158_corr5', 'alpha158_corr10', 'alpha158_corr20', 'alpha158_corr30', 'alpha158_corr60',
    'alpha158_cord5', 'alpha158_cord10', 'alpha158_cord20', 'alpha158_cord30', 'alpha158_cord60',
    'alpha158_cntp5', 'alpha158_cntp10', 'alpha158_cntp20', 'alpha158_cntp30', 'alpha158_cntp60',
    'alpha158_cntn5', 'alpha158_cntn10', 'alpha158_cntn20', 'alpha158_cntn30', 'alpha158_cntn60',
    'alpha158_cntd5', 'alpha158_cntd10', 'alpha158_cntd20', 'alpha158_cntd30', 'alpha158_cntd60',
    
    # RSI类因子（6个）
    'alpha158_sump6', 'alpha158_sump12', 'alpha158_sump24',
    'alpha158_sumd6', 'alpha158_sumd12', 'alpha158_sumd24',
    
    # 复合技术指标（11个）
    'alpha158_boll', 'alpha158_adx',
    'alpha158_k6', 'alpha158_k12', 'alpha158_k24',
    'alpha158_d6', 'alpha158_d12', 'alpha158_d24',
    'alpha158_j6', 'alpha158_j12', 'alpha158_j24',
    
    # 其他因子（3个）
    'alpha158_rank5', 'alpha158_rank10', 'alpha158_rank20'
]

print(f"因子总数: {len(alpha158_factors)}")
```

---

**文档版本**: 1.0  
**生成日期**: 2024年  
**状态**: ✅ 完整
