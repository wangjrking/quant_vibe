# 源码目录结构与防回退规则

## 目的

`quant/main` 只保留当前标准链路的入口和暂未完成包化的共享模块。一次性研究、历史复现和过期脚本不得继续堆放在根目录，以免误运行、GitHub 列表截断和依赖关系失真。

数据、DuckDB、报告和运行证据继续由 `quant/data_file/` 管理，不属于源码目录迁移范围。

## 目录职责

```text
quant/main/
  <root>/                 标准入口与受保护的共享模块
  core/                   可复用生产领域代码
  workflows/              L1-L8 工作流合同与调度
  tools/                  校验、治理和运维工具
  research/active/        正在进行的研究
  research/archive/       已结束研究的完整源码留档
  legacy/dated-scripts/   历史一次性脚本与兼容复现
  strategy_library/       策略定义、生产归档和研究档案
  tests/                  单元与集成测试
  docs/                   使用与治理文档
```

## 根目录规则

1. 根目录最多保留 172 个 Python 文件。
2. 仅允许标准 L1-L8 入口、共享生产模块和已登记的兼容基线模块留在根目录。
3. 新增的 `research_`、`tune_`、带日期 `build_`、`validate_`、`run_` 脚本必须进入 `research/`、`legacy/dated-scripts/` 或 `tools/`。
4. 根目录带日期文件必须登记在 `config/source_layout_policy_v1.json`；当前仅保留生产策略的受保护基线模块。

## 迁移要求

迁移一个文件或模块组时，必须同时完成：

1. 更新测试、复现清单、文档和受控路径引用。
2. 保持从仓库根目录执行时的导入兼容性，或明确将其标记为仅归档复现。
3. 运行相关单元测试、`tools/check_source_layout.py`、`tools/agent_governance_check.py` 和 `tools/standard_agent_architecture_check.py`。
4. 不得触碰生产数据、模型、信号或交易资产。

## 当前状态

截至 2026-08-30，根目录有 170 个 Python 文件。8 个带日期的历史研究和候选复现脚本已迁入 `research/archive/root-scripts/`；当前标准 L1-L8 入口未改变。

后续包化按数据、因子、模型、策略和治理工具分批进行。每批必须先完成依赖盘点和测试迁移，禁止使用全仓字符串替换。

## 自动检查

```powershell
cd D:\work\quant\quant_mcp\quant\main
D:\work\quant\quant_mcp\.venv\Scripts\python.exe tools\check_source_layout.py
```

检查会验证根目录数量、日期脚本例外、标准入口和关键目录。失败时应先修正文件位置或受控策略，不得绕过检查。
