# 📈 量化多因子投资项目

> 基于机器学习的股票收益率预测系统，通过多因子模型预测股票未来收益率，辅助量化投资决策。

---

## 📋 项目简介

本项目是一个工程化的量化投资系统，主要功能包括：

- **📊 数据采集**：从Tushare获取股票行情、财务、资金流向等数据
- **🔧 因子工程**：计算技术指标、基本面因子、Alpha因子等
- **🤖 模型训练**：使用XGBoost进行股票收益率预测
- **📉 因子分析**：使用Alphalens进行因子IC分析和分组回测
- **🔍 因子挖掘**：使用遗传编程挖掘新的Alpha因子
- **📧 消息通知**：自动发送预测结果邮件

---

## ✨ 功能特点

| 模块 | 功能 | 核心技术 |
|------|------|----------|
| **数据采集** | 多源数据获取与整合 | Tushare API、SQLite |
| **因子工程** | 技术指标与Alpha因子计算 | TA-Lib、自定义因子库 |
| **模型训练** | 股票收益率预测 | XGBoost、SHAP分析 |
| **因子分析** | 因子有效性检验 | Alphalens |
| **因子挖掘** | 自动因子发现 | GPlearn |
| **消息通知** | 结果推送 | SMTP邮件 |

---

## 🛠️ 技术栈

| 分类 | 技术 | 版本要求 |
|------|------|----------|
| **语言** | Python | 3.9 / 3.10 / 3.11（推荐 3.10） |
| **数据处理** | Pandas、NumPy | pandas>=1.5.0, numpy>=1.21.0 |
| **机器学习** | XGBoost、Scikit-learn、SHAP | xgboost>=1.6.0, scikit-learn>=1.0.0, shap>=0.40.0 |
| **因子分析** | Alphalens | alphalens>=0.4.0 |
| **因子挖掘** | GPlearn | gplearn>=0.4.0 |
| **数据采集** | Tushare | tushare>=1.2.0 |
| **数据库** | SQLite | Python 内置 |
| **技术指标** | TA-Lib、ta | ta>=0.10.0 |

---

## 📦 依赖模块详解

### 🐍 Python 环境要求

| 项目 | 说明 |
|------|------|
| **Python 版本** | 3.9, 3.10, 3.11 |
| **推荐版本** | 3.10.x（稳定且兼容性好） |
| **最低版本** | 3.9.0 |

### 📊 核心数据处理

| 模块 | 用途 | 版本要求 | 说明 |
|------|------|----------|------|
| **pandas** | 数据处理和分析 | >=1.5.0 | 数据读取、清洗、转换、聚合 |
| **numpy** | 数值计算 | >=1.21.0 | 高性能数组运算支持 |

### 🤖 机器学习

| 模块 | 用途 | 版本要求 | 说明 |
|------|------|----------|------|
| **xgboost** | 梯度提升算法 | >=1.6.0 | 核心预测模型 |
| **scikit-learn** | 机器学习工具 | >=1.0.0 | 数据预处理、模型评估 |
| **shap** | 可解释性分析 | >=0.40.0 | 特征重要性分析 |

### 📉 因子分析

| 模块 | 用途 | 版本要求 | 说明 |
|------|------|----------|------|
| **alphalens** | 因子回测 | >=0.4.0 | IC计算、分组收益率、信息比率 |

### 🔍 因子挖掘

| 模块 | 用途 | 版本要求 | 说明 |
|------|------|----------|------|
| **gplearn** | 遗传编程 | >=0.4.0 | 自动生成有效因子表达式 |

### 📡 数据采集

| 模块 | 用途 | 版本要求 | 说明 |
|------|------|----------|------|
| **tushare** | 金融数据接口 | >=1.2.0 | 股票行情、财务、资金流向数据 |

### 📈 技术指标

| 模块 | 用途 | 版本要求 | 说明 |
|------|------|----------|------|
| **ta** | 技术指标计算 | >=0.10.0 | MACD、RSI、ATR等 |
| **pandas-ta** | 扩展技术指标 | >=0.3.0 | 更丰富的技术指标（可选） |

### 💾 数据库

| 模块 | 用途 | 版本要求 | 说明 |
|------|------|----------|------|
| **sqlite3** | 本地数据库 | Python 内置 | 轻量级嵌入式数据库 |
| **sqlalchemy** | ORM框架 | >=1.4.0 | 数据库抽象层 |

### 🛠️ 其他工具

| 模块 | 用途 | 版本要求 | 说明 |
|------|------|----------|------|
| **tqdm** | 进度条 | >=4.62.0 | 数据处理进度显示 |
| **matplotlib** | 可视化 | >=3.5.0 | 图表绘制 |
| **pickle** | 对象序列化 | Python 内置 | 模型和数据对象保存 |

---

## 📋 安装命令

### 方式 1: 使用 pip 逐个安装

```bash
# 基础数据处理
pip install pandas>=1.5.0 numpy>=1.21.0

# 机器学习
pip install xgboost>=1.6.0 scikit-learn>=1.0.0 shap>=0.40.0

# 因子分析和挖掘
pip install alphalens>=0.4.0 gplearn>=0.4.0

# 数据采集
pip install tushare>=1.2.0

# 技术指标
pip install ta>=0.10.0

# 数据库
pip install sqlalchemy>=1.4.0

# 可视化和工具
pip install tqdm>=4.62.0 matplotlib>=3.5.0
```

### 方式 2: 使用 requirements.txt（推荐）

```bash
# 标准安装
pip install -r requirements.txt

# 使用国内源加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 升级所有依赖
pip install --upgrade -r requirements.txt
```

---

## 📁 项目结构

```
quant/main/
├── main.py                    # 项目主入口
├── config.json                # 配置文件
├── requirements.txt           # 依赖清单
├── README.md                  # 项目说明
│
├── ai_module.py               # AI模型模块（XGBoost训练与预测）
├── data_load_module.py        # 数据加载模块（Tushare API）
├── data_process_module.py     # 数据处理模块（因子工程）
├── database_module.py         # 数据库模块（SQL操作）
├── dl_model_module.py         # 深度学习模块（FTTransformer）
├── factor_mining_module.py    # 因子挖掘模块（遗传编程）
├── factor_analyzer_module.py  # 因子分析模块（Alphalens）
├── message_module.py          # 消息通知模块（邮件）
├── ta_compat.py               # 技术指标兼容层
│
├── check_ftt.py               # FTT模型检查脚本
├── test_ftt.py                # FTT模型测试脚本
├── stock_list.sql             # 股票列表SQL
├── alpha_factor_comparison.xlsx  # Alpha因子对比表
├── factor_comparison_summary.md  # 因子对比总结
└── __init__.py                # 包初始化
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd d:\work\quant\quant001\quant\main

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.json` 文件：

```json
{
  "project": {
    "name": "量化多因子投资项目",
    "version": "1.0.0",
    "description": "基于机器学习的股票收益率预测系统"
  },
  "database": {
    "url": "mysql+pymysql://root:password@localhost:3306"
  },
  "datasource": {
    "tushare_token": "your_tushare_token_here"
  },
  "filecatalog": {
    "file_url": "D:/work/quant/quant001"
  }
}
```

**配置说明**:
- `tushare_token`: 从 [Tushare官网](https://tushare.pro/) 获取
- `file_url`: 数据文件存储目录

### 3. 运行

```bash
# 方式1: 运行主程序
python main.py

# 方式2: 直接运行AI模块
python ai_module.py

# 方式3: 运行因子挖掘
python factor_mining_module.py

# 方式4: 运行因子分析
python factor_analyzer_module.py
```

---

## 📖 使用方法

### 基础使用

```python
from main import Run

# 创建并运行项目
run = Run()
```

### 数据加载

```python
from data_load_module import download_odb_data, get_pro

# 获取API实例
ts_pro = get_pro(ts_token='your_token')

# 下载数据
download_odb_data(
    stock_code_lst=['000001.SZ', '000002.SZ'],
    data_start_dt='20150101',
    data_end_dt='20250101',
    ts_token='your_token',
    data_file_url='./data_file'
)
```

### 因子工程

```python
from data_process_module import download_cdb_data

# 下载处理后的数据（含因子）
download_cdb_data(down_db=True, data_file_url='./data_file')
```

### AI预测

```python
from ai_module import download_pdb_data

# 运行AI预测
download_pdb_data(
    data_start_dt='20150101',
    data_test_dt='20230101',
    label='10d_yield_rate',
    type='reg',
    data_file_url='./data_file'
)
```

### 因子分析

```python
from factor_analyzer_module import analyze_factor, factor_summary

# 分析因子
result = analyze_factor(factor_data, 'my_factor')
summary = factor_summary(result['factor_returns'])
```

---

## 📊 核心模块说明

### 1. 数据加载模块 (`data_load_module.py`)

负责从Tushare获取各类金融数据：

- **行情数据**: `get_daily_data()` - 日线行情
- **财务数据**: `get_finan_data()` - 财务指标
- **资金流向**: `get_moneyflow()` - 资金流数据
- **筹码分布**: `get_stk_factor()` - 筹码因子
- **复权因子**: `get_adj_factor()` - 复权因子数据

### 2. 数据处理模块 (`data_process_module.py`)

负责因子计算和数据清洗：

- **技术指标**: MACD、RSI、ATR、EMA、DEMA、KAMA等
- **Alpha因子**: 国泰君安191因子库
- **自定义因子**: 基于价格、成交量的衍生因子
- **时间因子**: 年份、月份、季度、星期等

### 3. AI模型模块 (`ai_module.py`)

核心机器学习模块：

- **模型训练**: `model_assess()` - XGBoost训练与评估
- **特征分析**: `incre_fit()` - SHAP特征重要性分析
- **预测输出**: `download_pdb_data()` - 预测结果存储
- **自定义损失**: `huber_loss_obj()` - Huber损失函数

### 4. 因子挖掘模块 (`factor_mining_module.py`)

使用遗传编程自动发现新因子：

- **符号回归**: `SymbolicTransformer` - 自动生成因子表达式
- **特征选择**: 基于相关性筛选有效因子
- **因子验证**: 自动验证因子有效性

### 5. 因子分析模块 (`factor_analyzer_module.py`)

使用Alphalens进行因子分析：

- **IC分析**: 因子信息系数计算
- **分组回测**: 多分组收益率分析
- **风险调整**: 信息比率、夏普比率等

---

## ⚙️ 配置参数

### 时间参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_start_dt` | `20150101` | 训练数据开始日期 |
| `data_test_dt` | `20230101` | 测试数据开始日期 |
| `data_end_dt` | 当前日期 | 数据结束日期 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `label` | `10d_yield_rate` | 预测标签 |
| `type` | `reg` | 模型类型（reg/class） |
| `learning_rate` | `0.003` | 学习率 |
| `max_depth` | `3` | 树最大深度 |
| `n_estimators` | `5000` | 弱学习器数量 |

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `index_codes` | `['932000.CSI']` | 指数代码列表 |
| `exclude_bj` | `True` | 是否排除北交所股票 |

---

## 🔍 日志说明

日志文件存储在 `${file_url}/log/${date}.log`：

```log
2024-01-01 10:00:00 - INFO - 数据下载开始
2024-01-01 10:15:00 - INFO - AI模块启动
2024-01-01 11:00:00 - INFO - MSE: 0.00123456
2024-01-01 11:30:00 - INFO - 预测数据入仓完成
```

---

## 🧪 测试

```bash
# 运行FTT模型测试
python test_ftt.py

# 检查FTT模型
python check_ftt.py

# 测试数据加载
python -c "from data_load_module import get_pro; print('API连接成功')"
```

---

## ❓ 常见问题

### Q1: Tushare Token无效

**问题**: 运行时提示权限不足或Token无效

**解决方案**:
```bash
# 检查Token是否正确
# 访问 https://tushare.pro/user/token 获取有效Token
```

### Q2: 内存不足

**问题**: 处理大量数据时内存不足

**解决方案**:
```bash
# 减少数据范围
# 修改 data_start_dt 和 data_end_dt
```

### Q3: 依赖安装失败

**问题**: 某些依赖包安装失败

**解决方案**:
```bash
# 使用国内源
pip install package_name -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或降级版本
pip install xgboost==1.6.0
```

### Q4: 数据下载失败

**问题**: Tushare API调用失败

**解决方案**:
```bash
# 检查网络连接
# 检查Tushare积分是否足够
# 等待一段时间后重试
```

---

## 📝 贡献指南

1. **Fork 项目**
2. **创建功能分支**
   ```bash
   git checkout -b feature/xxx
   ```
3. **提交更改**
   ```bash
   git commit -m 'Add xxx feature'
   ```
4. **推送到分支**
   ```bash
   git push origin feature/xxx
   ```
5. **创建 Pull Request**

---

## 📄 许可证

MIT License

---

## 📧 联系方式

- **作者**: 王佳瑞
- **邮箱**: wangjiarui0808@163.com
- **项目地址**: https://github.com/yourusername/quant-project

---

## 🙏 致谢

感谢以下开源项目：

- [Tushare](https://tushare.pro/) - 金融数据接口
- [XGBoost](https://xgboost.readthedocs.io/) - 梯度提升算法
- [SHAP](https://shap.readthedocs.io/) - 可解释性分析
- [Alphalens](https://alphalens.readthedocs.io/) - 因子分析
- [GPlearn](https://gplearn.readthedocs.io/) - 遗传编程
- [TA-Lib](https://mrjbq7.github.io/ta-lib/) - 技术分析库

---

## 📌 更新日志

### v1.0.0 (2024-01-01)
- ✅ 基础数据采集功能
- ✅ 多因子工程模块
- ✅ XGBoost模型训练
- ✅ 因子分析功能
- ✅ 遗传编程因子挖掘
- ✅ 消息通知功能

---

**版本**: 1.0.0  
**最后更新**: 2024年1月  
**文档状态**: ✅ 完整

---

⭐ **如果这个项目对你有帮助，请给个Star！** ⭐
