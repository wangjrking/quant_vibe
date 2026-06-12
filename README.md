# Quant Vibe 多因子量化项目

这是一个面向 A 股的多因子量化研究与交易信号项目，包含数据更新、因子加工、模型训练、选股、回测、掘金信号导出和报表生成等流程。

项目已经改成以仓库根目录为基准的相对路径，新用户 clone 后配置依赖和 Token 即可运行，不需要修改本机绝对路径。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/wangjrking/quant_vibe.git
cd quant_vibe
```

### 2. 创建环境

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果是在 Linux/macOS 上运行，虚拟环境激活命令通常是：

```bash
source .venv/bin/activate
```

### 3. 配置 Token

推荐通过环境变量配置 Tushare Token，避免把个人密钥提交进仓库：

```powershell
$env:TUSHARE_TOKEN="your_token_here"
```

也可以复制 `config.example.json` 为 `config.json` 后手动填写：

```bash
copy config.example.json config.json
```

默认配置使用相对路径：

| 配置项 | 默认位置 |
| --- | --- |
| 配置文件 | `config.json` |
| 数据目录 | `data_file/` |
| 日志目录 | `log/` |
| 报表目录 | `data_file/reports/` |
| 掘金策略目录 | `juejin_strategies/` |

可以通过环境变量覆盖数据目录，例如改到仓库内的 `my_data/`：

```powershell
$env:QUANT_DATA_DIR="my_data"
```

## 常用命令

```bash
# 主流程：数据下载、因子加工、训练预测
python main.py

# 更新因子数据
python run_cdb_update.py

# 训练模型并写入预测表
python run_pdb_update.py

# 每日增量流程（Windows PowerShell）
.\run_daily_strategy.ps1

# 导出股票池
python export_stock_pool.py

# 生成掘金策略报告
python generate_juejin_strategy_reports.py
```

掘金相关脚本默认读取 `juejin_strategies/` 下的相对目录。新用户需要把自己的掘金策略 `main.py` 放到对应目录，或通过命令行参数传入策略目录。

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `project_paths.py` | 统一解析项目根目录、配置文件和数据目录 |
| `config.json` | 本地默认配置，Token 建议用环境变量覆盖 |
| `config.example.json` | 可提交、可分享的配置模板 |
| `data_file/` | 本地数据、SQLite 数据库、信号和报表输出目录，默认不提交 |
| `log/` | 运行日志目录，默认不提交 |
| `juejin_strategies/` | 本地掘金策略目录，默认不提交 |
| `tests/` | 项目测试 |

## 开箱即用约定

- 不依赖任何本机固定绝对路径。
- 所有默认数据输出都落在仓库内的 `data_file/`。
- `config.json` 中的 `datasource.tushare_token` 可以为空，运行时优先读取 `TUSHARE_TOKEN`。
- 大体积数据、日志、缓存和本地掘金策略不进入 Git。
- 如果需要把数据放到仓库外，用 `QUANT_DATA_DIR` 覆盖即可。

## 注意事项

本项目用于量化研究和策略验证，不构成任何投资建议。实盘前请自行完成数据核验、回测复现、交易成本评估、风控设置和小资金验证。
