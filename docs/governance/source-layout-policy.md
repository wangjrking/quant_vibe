# 源码目录架构与防回退规则

## 目的

`quant/main` 曾长期混放正式入口、共享模块、一次性实验和历史脚本，导致根目录难以阅读、GitHub 文件列表截断，也增加了误运行旧脚本的风险。本规则将“当前可运行主线”和“历史可追溯代码”分开管理。

本规则只治理源码与文档目录；市场数据、DuckDB、报告、运行证据仍按 `quant/data_file/` 的资产与 runtime 规则管理。

## 目标结构

```text
quant/main/
  <root>                 当前正式入口、既有共享模块与受保护兼容入口
  core/                  新增可复用生产领域代码
  workflows/             L1-L8 工作流合同和调度定义
  tools/                 只读校验、迁移、治理和运维工具
  config/                可版本化的配置与治理策略
  research/              正在进行的研究
    archive/             已结束研究的完整源码留痕
  legacy/                非主线兼容与历史复现
    dated-scripts/       带日期的一次性执行、构建、调参、验证脚本
  strategy_library/      策略定义、生产归档和研究档案
  execution_gateway/     非执行态交易交付适配层
  tests/                 单元与集成测试
  docs/                  使用与治理文档
```

## 根目录准入

根目录不是通用脚本库，只允许以下三类文件：

1. L1-L8 标准工作流直接入口及其严格必要的兼容入口。
2. 被多个标准入口导入、尚未完成包化迁移的共享模块。
3. 被测试、策略归档、复现清单或正式配置按原路径直接引用的受保护历史文件。

根目录 Python 文件上限为 **220**。带完整日期的 Python 文件默认不得新增到根目录；现有例外必须登记在 `config/source_layout_policy_v1.json`。任何新增例外都需要在同一变更中说明引用方与退役计划。

## 新文件唯一落点

| 类型 | 必须放置位置 |
| --- | --- |
| 可复用生产代码 | `core/<domain>/` |
| L1-L8 工作流合同 | `workflows/` |
| 校验、扫描、治理、运维工具 | `tools/` |
| 正在进行的研究 | `research/<topic>/` |
| 已结束研究 | `research/archive/` |
| 一次性或带日期的历史脚本 | `legacy/dated-scripts/` |
| 测试 | `tests/` |
| 数据、报告、DuckDB、运行证据 | `quant/data_file/`，不得进入源码根目录 |

不得为“方便临时运行”在根目录新增 `research_`、`tune_`、`run_YYYYMMDD`、`build_YYYYMMDD` 或 `validate_YYYYMMDD` 脚本。

## 迁移策略

现存共享模块不进行无边界的大搬家。只有当一个领域模块完成以下条件，才迁入 `core/<domain>/`：

1. 其全部静态和动态导入方已盘点。
2. 标准工作流、测试和正式 manifest 的路径已同步更新。
3. 迁移后通过对应测试、`check_source_layout.py` 与治理检查。
4. 生产资产、数据、模型、信号和交易执行均未被触发。

这使目录可持续变干净，而不会为了视觉整齐破坏当前主线。

### 后续包化路线

当前标准 L1-L8 入口的静态依赖闭包约为 42 个根目录模块；另有少量历史受保护文件。其余根目录文件应按下列顺序逐步收口，单次只迁移一个领域并完成对应测试：

| 阶段 | 目标目录 | 迁移内容 | 完成标准 |
| --- | --- | --- | --- |
| 1 | `core/data/` | L1/L2 路由、原始表、复权和资产路径共享模块 | L1/L2 target-date 测试与路由合同通过 |
| 2 | `core/factor/` | L3 合同、GTJA、标签和 writer lease 共享模块 | L3 preflight、写入旁路扫描通过 |
| 3 | `core/model/` | formal manifest、模型路由、预测与标签共享模块 | L4 postwrite 与 manifest 绑定测试通过 |
| 4 | `core/strategy/` | 组合、策略资产路由、L5/L6/L7 同步模块 | L5-L7 合同与 pending-only 测试通过 |
| 5 | `tools/` | 一次性修复、审计辅助、导入导出和运维脚本 | 工具不被标准 workflow 或生产 manifest 引用 |

根目录的下一目标是少于 120 个 Python 文件；在未完成上述某一领域的依赖盘点前，不以“清理”为理由移动共享模块。

## 自动门禁

执行以下命令：

```powershell
cd D:\work\quant\quant_mcp\quant\main
D:\work\quant\quant_mcp\.venv\Scripts\python.exe tools\check_source_layout.py
```

门禁会验证根目录文件数量、日期脚本例外、L1-L8 入口和关键目录。任何失败都应先调整文件位置或更新受审策略，不能绕过检查。

## 当前状态

2026-08-30 整理后，根目录有 194 个 Python 文件；历史研究脚本位于 `research/archive/root-scripts/`，带日期的一次性脚本位于 `legacy/dated-scripts/`。此状态是后续新增代码的基线，而不是再次堆积脚本的许可。
