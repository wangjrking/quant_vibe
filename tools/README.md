# 工具目录

## 定位

本目录保存项目辅助工具。工具不得默认触发业务链路。

## 当前治理工具

```text
agent_governance_check.py
```

用途：

- 检查智能体文档包是否齐全。
- 检查智能体工作区和实验代码区说明是否存在。
- 检查资产注册表 JSON 是否可解析。
- 检查工作流监控模板是否存在。

运行方式：

```text
D:\work\quant\quant_mcp\.venv\Scripts\python.exe quant\main\tools\agent_governance_check.py
```

## 当前结构治理工具

```text
repo_noise_report.py
runtime_noise_report.py
runtime_loose_files_report.py
structure_governance_dispatch.py
prepare_structure_manifest_drafts.py
```

用途：

- `repo_noise_report.py` 只读扫描 `quant/main` 根目录中疑似研究、调参、验证、归档辅助脚本。
- `runtime_noise_report.py` 只读扫描 `quant/data_file/runtime` 根目录中长期保留、归档复核、回收站复核和人工复核候选。
- `runtime_loose_files_report.py` 只读扫描 `quant/data_file/runtime` 根目录散落文件，单独输出 loose files 认领包。
- `structure_governance_dispatch.py` 基于前两个报告生成按智能体拆分的认领包和 runtime 复核包。
- `prepare_structure_manifest_drafts.py` 基于已确认的治理分组生成 manifest 草稿，供审计和后续执行前置使用。
- 这些工具只生成治理候选清单，不移动、不删除、不执行业务链路。

运行方式：

```text
D:\work\quant\quant_mcp\.venv\Scripts\python.exe quant\main\tools\repo_noise_report.py --write-report quant\data_file\reports\project_structure_governance_YYYYMMDD\quant_main_noise_report.json
D:\work\quant\quant_mcp\.venv\Scripts\python.exe quant\main\tools\runtime_noise_report.py --write-report quant\data_file\reports\project_structure_governance_YYYYMMDD\runtime_noise_report.json
D:\work\quant\quant_mcp\.venv\Scripts\python.exe quant\main\tools\runtime_loose_files_report.py --write-report quant\data_file\reports\project_structure_governance_YYYYMMDD\runtime_loose_files_report.json
D:\work\quant\quant_mcp\.venv\Scripts\python.exe quant\main\tools\structure_governance_dispatch.py --report-dir quant\data_file\reports\project_structure_governance_YYYYMMDD
D:\work\quant\quant_mcp\.venv\Scripts\python.exe quant\main\tools\prepare_structure_manifest_drafts.py --report-dir quant\data_file\reports\project_structure_governance_YYYYMMDD
```

## 当前交易辅助工具

```text
qmt_holdings_api.py
```

用途：

- 通过本机 `QMT/xtquant` 官方 Python API 读取当前资金和持仓。
- 输出持仓摘要，供交易智能体生成人工下单模板或核对模拟盘/实盘仓位。
- 输出 JSON 诊断结果，定位 `xtdata`、`XtQuantTrader`、`qmttools` 哪一层失败。

运行方式：

```text
D:\work\quant\quant_mcp\.venv\Scripts\python.exe quant\main\tools\qmt_holdings_api.py --account-id <QMT账户ID>
```

诊断方式：

```text
D:\work\quant\quant_mcp\.venv\Scripts\python.exe quant\main\tools\qmt_holdings_api.py --account-id <QMT账户ID> --format json
```
