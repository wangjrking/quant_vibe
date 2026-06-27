## 2026-06-22 ST / Risk-Warning Stock Hard Gate

This is a mandatory cross-agent rule for data, factor, model, strategy,
backtest, audit, and trading work.

- Any strategy signal, candidate pool, formal backtest, production archive, or
  trading handoff that claims `exclude_st=true` must filter ST / risk-warning
  stocks by the governed L2 fields, not by display name only.
- The required source of truth is
  `data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA` on the relevant
  `trade_date` / `signal_date` / `buy_date`.
- A stock is ST / risk-warning if any of the following holds:
  `ST_TYPE` is not null/empty/`0`/`0.0`/`None`/`NONE`; `ST_TYPE_name` indicates
  a risk-warning board; `name` starts with `ST` or `*ST`.
- Name-only checks such as `name NOT LIKE 'ST%'` are insufficient because the
  project data can contain ordinary display names while `ST_TYPE='ST'`.
- Before reporting a strategy as current-best, investable, production-ready, or
  suitable for retail communication, strategy-agent must run an ST audit by
  joining the signal rows to `STOCK_DAILY_DATA` on `(stock_code, signal_date)`
  and, when available, `(stock_code, buy_date)`.
- trading-agent must refuse formal delivery if the signal file contains any ST
  / risk-warning row after this join, unless the user explicitly approves an
  ST-allowed experimental run and the output is marked non-production.
- data-ingestion-agent and data-integration-agent must preserve and audit
  `stock_st` / `ST_TYPE` / `ST_TYPE_name` coverage. Missing or stale ST fields
  are blocking data-quality issues for downstream signal generation.

## 2026-06-24 策略生产准入强弱分层口径

这是策略侧判断“当前最优”“投产候选”“生产准入通过”“交交易智能体”的统一口径。

- 收益类指标是弱准入项，包括年化收益、累计收益、胜率、交易次数、平均持仓率、Calmar 和 Sharpe 的目标水平。收益指标用于排序和优先级判断，不能单独放行生产，也不能因单一收益指标未达到目标而直接否决仍具备稳健性的候选。
- 敏感性和持续性是强准入项，包括参数邻域稳定性、时间切片稳定性、低路径依赖、近期空仓启动表现、连续账户同区间切片对比、未来信息排除、硬过滤审计、掘金复现和生产归档完整性。
- 风格暴露和风格漂移是必披露风险提示项，不作为单独阻断生产准入的强门槛；只有当它们同时暴露未来信息、参数敏感、低路径依赖、时间切片失效、硬过滤失效或样本过窄等问题时，才按对应强准入项处理。
- 强准入项任一不通过，策略不得标记为生产准入通过；高年化只能保留为研究证据、路径依赖候选或用户指定生产覆盖版本。
- 策略报告必须分别列出弱准入排序结果和强准入通过/不通过项，避免把“收益高”与“可持续投产”混为同一个结论。

## 2026-06-24 风格暴露和风格漂移提示口径

这是策略侧对“当前最优”“投产候选”“生产准入通过”“交交易智能体”的必披露风险提示规则。该规则不单独阻断生产准入。

- 策略不得只凭收益和回撤准入，必须审计持仓和收益是否明显暴露于单一风格，例如小市值、低流动性、高换手、高波动、单一行业、单一板块、次新股或极端价格区间。
- 风格暴露审计必须使用信号日或买入日当时可得字段，不得用最新市值、最新行业、最新 ST 状态、最新 ATR 或最新流动性回填历史样本。
- 风格漂移审计必须比较全周期、最近 20/60/120 个交易日、按年或滚动窗口、低路径依赖启动窗口和参数邻域中的主要风格分布。
- 如果参数微调后策略从一种主导风格跳到另一种主导风格，且收益、回撤或信号覆盖明显不稳定，应按参数邻域稳定性强准入项处理，而不是仅按风格漂移处理。
- 如果收益主要来自单一行业、单一板块、极端市值、极端流动性、极端波动、极端换手或次新股暴露，必须标注为风格暴露提示；除非同时触发滚动窗口失效、参数邻域失效、未来信息或样本过窄问题，否则不得仅因风格集中度否决投产。
- 策略报告必须披露主要风格维度的持仓分布、收益贡献、最大集中度、近期窗口差异和参数邻域一致性。

## 2026-06-23 低路径依赖策略准入门槛

这是策略侧对“当前最优”“投产候选”“可投产”“新账户交接”结论的强制准入规则。

- 策略不得只凭全周期连续回测年化收益准入。策略智能体必须验证该收益是否依赖
  特定历史启动路径。
- 必须保留的证据包括：全周期空仓启动回测、多个近期锚点附近的空仓启动回测、
  连续账户同区间 NAV 切片，以及在策略存在持仓状态依赖卖出时的 warm-up /
  影子状态验证。
- 近期相邻空仓启动检查在数据允许时至少覆盖 3 个近期锚点，每个锚点应包含
  若干相邻可交易起点；如果最新窗口无法满足，必须明确写出限制。
- 如果近期空仓启动显著弱于连续账户同区间切片，该策略必须标记为路径依赖研究版
  或连续账户型候选，不得把全周期年化收益描述为新资金账户的预期结果。
- 策略报告必须同时披露：全周期年化、近期空仓启动年化范围、近期空仓启动最小
  和中位年化、连续账户同区间年化、warm-up 结果（如已测试）、Sharpe、最大回撤、
  交易次数和证据路径。
- 低路径依赖候选应优先采用能快速收敛到当日组合的规则，例如更短持仓周期、
  更少状态化卖出锁定、更少每日卖出瓶颈、明确的固定权重或等权规则。但这些
  设计选择本身不足以准入；准入仍以掘金回测和启动稳定性证据为准。
- 未通过该门槛的策略可以保留在研究归档中，但不得提升为唯一 L5 生产策略，
  也不得交给交易智能体作为新账户默认策略。

# AGENTS.md

## 跨线程协同发送口径

所有智能体必须区分“通信层留痕”和“真实线程发送”：

- 写入 `quant/data_file/runtime/agent_memory/collaboration_requests.jsonl`、`agent_notes/` 或其他通信层文件，只代表已记录为待处理上下文，不代表已发送到目标智能体会话。
- 用户明确要求“发给某智能体”“通知审计智能体”“发到会话/线程”时，必须使用真实跨线程发送工具，不能只写通信层文件。
- 如果当前只写了通信层而尚未发送真实线程，回复必须明确写成：“已写通信层，尚未发送到目标线程”。
- 只有真实线程消息已发送并确认进入目标智能体会话后，才能表述为“已发给审计智能体”“已通知架构师智能体”“已发到会话”。
- 审计送审、正式资产变动复核、权限申请、阻塞升级和投产影响事项，不能用通信层留痕替代线程级分派或送审。

## 跨智能体审计闭环

凡是其他智能体在线程中向审计智能体发起的送审、复核、验收或可交接判断请求，审计完成后必须把审计回执回传到原发起线程。

- 不能只在当前线程回答。
- 不能只写 `communication layer`、`agent_notes/` 或审计报告路径。
- 只有原发起线程收到审计回执后，才算该次跨智能体审计闭环完成。
- 审计已完成但尚未回传原发起线程时，只能表述为“审计已完成，线程回执未完成”。

审计回执至少包含：

- 是否通过。
- 风险等级。
- 关键证据路径。
- 是否可放入正式资产。
- 是否可交接下游正式链路。
- 如不通过，责任归口和下一步建议。
- 审计记录路径。
- 线程闭环状态。

## 全体智能体交易日口径

所有智能体在处理数据、因子、标签、模型、策略、信号、回测、交易、审计或架构评审时，必须先明确交易日口径。

- 禁止把当前自然日直接当成交易日。
- 禁止把“今天”“昨天”“明天”“最新”直接写成结论，必须转换为明确的 `YYYYMMDD`。
- 必须区分当前自然日、目标交易日、最新已完成交易日、信号日期、执行交易日和回刷窗口。
- 最新已完成交易日必须同时满足交易日历开市和关键上游资产审计通过。
- 任一关键源接口或上游资产未到齐时，不能用旧交易日数据冒充目标交易日或最新信号。
- 详细规范见 `.codex/agent_packages/trading-day-rules.md`。

## 2026-06-18 GTJA Official Formula Governance

Current GTJA Alpha191 reproduction governance for the maintained L3 factor
chain:

- Maintained GTJA formula source: `quant/main/alpha191.py` and
  `quant/main/alpha191.txt`
- Maintained executor: `quant/main/gtja_alpha_official_core.py`
- Default maintained `RANK(x)` semantics: full-market cross-sectional rank by
  `trade_date`, not legacy raw-value passthrough
- `alpha030` currently uses locally derived `MKT` / `SMB` / `HML` inputs from
  project data, so it is a project-level approximation / governance exception
  (`项目内近似` / `治理例外`) rather than a claim of strict equivalence to an
  external official FF3 source
- Maintained daily/incremental factor updates must call the official executor
  with `cross_sectional_rank_mode="rank"`
- Legacy identity-rank experiments or smoke assets must be explicitly marked
  `raw_gtja_identity` / `legacy` and must not overwrite the maintained
  production contract

## 2026-06-18 Industry Field Governance

Industry-field governance for the current L3 default feature chain:

- The default industry field is `industry` text.
- Legacy `data_file/stock_factor_data.parquet.industry_encode` is **not** a
  trusted stable industry-classification code and must not be reused as the
  default industry encoding contract for the new chain.
- `stock_factor_data.parquet` remains a historical compatibility / audit /
  rebuild evidence asset only; its `industry_encode` must not be assumed to be
  a stable one-to-one mapping for current production training or prediction.
- If a stable numeric industry code is needed in the new chain, it must come
  from a separately governed mapping asset, not from inferring or copying the
  legacy mixed wide table.
- Current governed mapping asset for the production feature chain:
  `data_file/runtime/production_factor_industry_encode_mapping.json`
- If `production_factor_parts/` contains `industry_encode`, that column is only
  valid when it is derived from the governed mapping asset above. When this
  mapping asset is absent or not explicitly governed, callers must fall back to
  `industry` as the default industry field.

## 2026-06-17 P3 Governance: Legacy `stock_factor_data.parquet`

Audit-agent confirmed that the historical compatibility wide table
`data_file/stock_factor_data.parquet` contains future/label columns. It is
therefore **not** the default production feature input for the current layered
workflow.

Current L3 delivery contract:

- Production features: `data_file/production_factor_parts/`
- Training labels: `data_file/prediction_label_parts/`
- Historical compatibility / audit / rebuild evidence only:
  `data_file/stock_factor_data.parquet`

Any script, report, or agent handoff that uses `stock_factor_data.parquet` for
model training must explicitly treat it as a legacy mixed wide table and remove
future/label columns before feature use. New production training or prediction
work must not read it as the default feature source.

As of 2026-06-17, model-side legacy mixed asset usage is also blocked by
default unless `QUANT_ALLOW_LEGACY_MODEL_ASSET_CHAIN=1` is set explicitly for
rollback, migration, or legacy reproduction.

## 2026-06-17 L4 -> L5 Formal Manifest Contract

Current production L5 reads must follow this contract:

- A formal prediction manifest must have `approval_status=approved_for_l5`.
- It must declare a non-legacy L4 prediction source via explicit `db_path` and
  `table`.
- If the manifest points to `odb.db`, sets `legacy_source=true`, or uses an
  asset role beginning with `legacy`, it is not a current default L5 input.
- `run_production_tasks.py` and `export_gm_signals.py` should only consume the
  approved formal manifest path in the standard workflow.
- Legacy prediction tables under `odb.db.stock_predict_data_*` remain available
  only for rollback, audit comparison, or explicit legacy reproduction.

## 2026-06-17 L1 Path Contract

Current authoritative L1 path contract:

- Unified Python: `D:\work\quant\quant_mcp\.venv\Scripts\python.exe`
- Main config: `D:\work\quant\quant_mcp\quant\main\config.json`
- L1 standard data root: `D:\work\quant\quant_mcp\quant\data_file`
- Standard stock pool: `D:\work\quant\quant_mcp\quant\data_file\stock_pool_all_a.csv`
- Split raw DB root: `D:\work\quant\quant_mcp\quant\data_file\raw_table_dbs\`
- Legacy raw DB fallback: `D:\work\quant\quant_mcp\quant\data_file\odb.db`

## 2026-06-17 Research Agent Contract

Current governance for the new research role:

- `research-agent` is a research-and-recommendation agent, not an execution
  agent.
- It may read project docs, audits, strategy archives, and external research
  materials, then summarize candidate improvements for factors, models,
  strategies, trading execution, and risk controls.
- It must report recommendations to the commander/orchestrator before any
  implementation work is assigned.
- It must not directly ingest data, compute factors, train models, generate
  predictions, generate signals, run backtests, or modify production assets.
- Any recommendation that would change production defaults, trigger retraining,
  re-prediction, signal publication, or live-trading behavior still requires
  commander review and user approval.

Governance rules:

- L1 maintained entrypoints must resolve data assets from the standard data root,
  not from `quant\main\data_file`.
- `project_paths.py` remains the authoritative helper for maintained L1 path
  resolution; code/config assets and data assets should not share the same
  default base blindly.
- `run_all_a_raw_update.py` is currently the stock-pool-driven incremental raw
  ingestion entry. A separate trade-date-full ingestion mode is an approved next
  architecture item for data-ingestion-agent, not implied by the current stock-
  pool incremental path.

## 当前新增硬约束：预测标签表单独存放

- 原始因子表保留 raw 计算结果和 future label 原始辅助列，用于审计和重建。
- 投产因子表只允许放训练/预测特征：raw 可用因子 + 按交易日全市场标准化后的因子；禁止放 `post*`、`*_yield_rate`、`*_tag`、`executable_*_return` 等未来标签列。
- 预测标签表必须单独存放，用于模型训练时按 `trade_date, stock_code` 与投产因子表 join。
- 当前预测标签表目录为 `data_file/prediction_label_parts/`，生成脚本为 `quant/main/build_prediction_label_parts.py`。
- 当前标签表包含已有未来收益标签和可执行开盘收益标签；可执行标签统一按 `post_open` 买入、对应未来开盘价卖出，并扣买入佣金 0.0003、卖出佣金 0.0003、印花税 0.0005、滑点 0.001。
- `top05/top10/top20` 等截面分类标签不得按股票分片计算；如需使用，必须按 `trade_date` 读取全 A 后做全市场排名，单独生成分类标签表。
- 当前投产因子宽表目录为 `data_file/production_factor_parts/`，它指向 `data_file/production_factor_parts_clean_20260616/`；该版本已排除 `cyq_perf`、`limit_list_data`、`stock_st` 等源头受限字段，GTJA Alpha 公式中的 `RANK(x)` 明确不做截面排序，直接使用 `x` 原值合并为 `gtja_alpha001` ~ `gtja_alpha191`。
- 如需恢复 GTJA 全市场 RANK 版，必须另起目录、另写 schema/report，并在训练报告中明确标记，不能覆盖当前 `production_factor_parts` 默认目录。

## 数据资产清单和分层边界

本节用于明确审计、数据接入、底表、因子、模型、策略和部署之间的数据资产边界。后续任何补数、重建、训练、预测、出信号、回测或投产归档，都必须先判断目标资产属于哪一层，再由对应智能体处理；不得跨层直接写入。

### L0 配置、运行状态和通信层

Python 环境硬约束：

- 本项目所有智能体、脚本、审计、数据接入、数据整合、因子、模型、策略、回测和部署任务，必须使用同一个项目级 Python 虚拟环境。
- 统一虚拟环境位置为项目根目录：`D:\work\quant\quant_mcp\.venv\`。
- 统一解释器路径为：`D:\work\quant\quant_mcp\.venv\Scripts\python.exe`。
- 当前统一 Python 版本为：`Python 3.9.23`。
- 统一依赖清单为：`D:\work\quant\quant_mcp\quant\main\requirements.txt`。
- Windows 上不得使用 `C:\Users\wangj\AppData\Local\Microsoft\WindowsApps\python.exe` 这类 Microsoft Store 占位符解释器运行项目任务。
- 不得使用 Python 3.13 创建本项目运行环境；当前依赖中的 `alphalens` 等库与 Python 3.13 不兼容。
- 各智能体不得自行创建新的 `.venv`、conda 环境或临时 Python 环境；如环境缺失或依赖不完整，必须反馈指挥官智能体，由指挥官授权环境初始化或修复。
- 所有可复现命令、任务日志和投产归档必须记录实际使用的 Python 解释器路径和关键依赖版本。

资产范围：

- 配置模板：`.env.example`、`config.example.json`、`config/production_tasks.example.json`。
- 本地私有配置：`.env`、`config.json`、`config/production_tasks.json`，不得提交真实 token、账号、密钥或本机个人路径。
- 运行状态：`data_file/runtime/`。
- 多智能体通信：`data_file/runtime/agent_memory/`。
- 运行锁建议目录：`data_file/runtime/locks/`。

边界：

- 只能保存配置、状态、锁、任务上下文和已确认决策。
- 不得保存真实 Tushare Token、掘金账号密钥、未脱敏登录信息或未经主管确认的投资结论。
- L0 变更不得直接触发 L1-L7 业务链路；触发链路必须由指挥官智能体授权分派。

### L1 原始数据层

资产范围：

- 原始 parquet 第一落点：`data_file/daily_data.parquet`、`data_file/daily_index_data.parquet`、`data_file/stk_factor.parquet`、`data_file/moneyflow.parquet`、`data_file/limit_list_data.parquet`、`data_file/cyq_perf.parquet`、`data_file/adj_factor.parquet`、`data_file/stock_st.parquet`、`data_file/index_daily.parquet`、`data_file/stock_basic_data.parquet`、`data_file/finan_data_season.parquet`、`data_file/finan_data_year.parquet`、`data_file/top_list.parquet`、`data_file/ths_hot.parquet`、`data_file/dc_hot.parquet`。
- SQLite 原始表默认资产：`data_file/raw_table_dbs/[table].DB`，每张 raw table 单独一个 DB 文件。
- Legacy 兼容源：`data_file/odb.db` 中的 raw tables 仅作为历史兼容、回滚和迁移审计源保留，不再作为 L1 默认写入目标。
- 原始数据审计证据：`data_file/reports/raw_data_audit_<trade_date>_before_fill.csv`、`data_file/reports/raw_data_audit_<trade_date>_after_fill.csv`、`data_file/reports/raw_data_fill_<trade_date>_summary.*`、`data_file/reports/tushare_probe_<trade_date>.*`。
- 原始数据临时补洞分片：`data_file/backfill_parts/`、`data_file/raw_update_smoke/`。

边界：

- 数据接入智能体只负责 L1：外部源探测、原始 parquet 写入、原始表同步和原始数据审计。
- L1 不得计算 `STOCK_DAILY_DATA`，不得计算因子，不得生成标签、预测、信号或回测结论。
- 原始 parquet 是补数第一落点；原始 parquet 审计未通过，不得同步 SQLite 原始表。
- SQLite 原始表同步必须默认写入 `raw_table_dbs/[table].DB`；同步完成后只能交接给 L2，不得直接跳到 L3-L7。
- 关键源接口未到齐时，状态应标记为 `source_not_ready`，不得用旧数据冒充目标交易日。

### L2 综合底表层

负责智能体：`data-integration-agent`（数据整合智能体）。

资产范围：

- 当前 L2 默认资产：`data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA`。
- Legacy 兼容源：`data_file/odb.db::STOCK_DAILY_DATA` 仅作为迁移源、回滚和审计比对资产保留，不得作为新链路默认读取目标。
- 综合底表审计报告：`data_file/reports/stock_daily_*`、`data_file/reports/data_quality_audit_*` 或同等命名的底表质量报告。

边界：

- L2 输入只能来自已审计通过的 L1 原始 parquet / `raw_table_dbs/[table].DB` 原始表。
- `STOCK_DAILY_DATA` 负责统一行情、估值、资金流、筹码、涨跌停、ST、指数和复权因子。
- 若策略要求前复权，`STOCK_DAILY_DATA.open/high/low/close` 必须统一为前复权口径。
- L2 不得生成因子宽表、预测表、买卖信号或回测收益结论。

### L3 因子和标签层

资产范围：

- 原始因子分片：`data_file/raw_factor_by_stock_parts/`。
- 原始因子烟测分片：`data_file/raw_factor_by_stock_parts_smoke/`。
- 标准化因子分片：`data_file/standard_factor_by_date_parts/`、`data_file/standard_factor_by_date_parts_fast40/`。
- 历史兼容因子宽表：`data_file/stock_factor_data.parquet` 及明确标注的备份文件。
- 当前投产因子宽表：`data_file/production_factor_parts/`，当前指向 `data_file/production_factor_parts_clean_20260616/`。
- 因子生产临时/烟测目录：`data_file/factor_rebuild_parts/`、`data_file/factor_rebuild_smoke/`、`data_file/production_factor_smoke/`、`data_file/production_factor_raw_gtja_identity_smoke/`、`data_file/gtja_full_market_only_parts/`、`data_file/gtja_only_smoke/`。
- 预测标签表：`data_file/prediction_label_parts/`。
- 标签烟测目录：`data_file/prediction_label_smoke/`。
- 因子和标签 schema / audit：`data_file/reports/*factor*schema*.json`、`data_file/reports/*factor*columns*.csv`、`data_file/reports/*gtja*rank*audit*.json`、`data_file/reports/*prediction_label*schema*.json`。

边界：

- L3 输入只能来自已审计通过并覆盖目标区间的 L2。
- 原始因子表可以保留 raw 计算结果和 future label 原始辅助列，用于审计和重建。
- 投产因子表只允许放训练/预测特征，禁止放 `post*`、`*_yield_rate`、`*_tag`、`executable_*_return` 等未来标签列。
- 预测标签必须单独存放在 `prediction_label_parts/`，训练时按 `trade_date, stock_code` 与投产因子表 join。
- 按股票维度计算的 raw factor 与按交易日维度计算的标准化 factor 必须分目录、分报告；不得在股票批次内生成最终全市场 rank。

### L4 模型训练和预测层

资产范围：

- 特征筛选结果：`data_file/selected_features*.json`、`data_file/rolling_fold_features*/`。
- 特征 IC 证据：`data_file/feature_ic_scores*.csv`。
- 训练摘要：`data_file/rolling_train_summary*.csv`。
- 模型解释或重要性：`data_file/feature_importance.csv`、`data_file/train_feature_importance.csv`、`data_file/test_feature_importance.csv`、`data_file/shap_values.pkl`。
- 预测表和模型证据应优先进入策略归档目录：`strategy_library/production/<strategy_id>/predictions/`、`strategy_library/production/<strategy_id>/models/`、`strategy_library/production/<strategy_id>/feature_selection/`。

边界：

- L4 输入只能来自 L3 的投产因子表和单独标签表。
- 模型训练默认 `expanding2010`，不得把非 `expanding2010` 结果称为正式候选或投产版本。
- L4 可以输出预测分和模型证据，但不得制定交易规则、生成正式买卖信号或声称策略收益达标。
- 如原版模型 checkpoint 缺失，重训预测不得冒充原版投产信号。

### L5 策略信号层

资产范围：

- 正式/候选信号：`data_file/production_signals/`、`data_file/gm_signals*.csv`、策略归档中的 `signals/`。
- 每日信号摘要：`data_file/daily_strategy_summary.json`、`data_file/daily_signal_summary*.json`。
- 策略规则和注册表：`strategy_library/registry.json`、`strategy_library/production/<strategy_id>/trading_rules.json`。
- 平台执行交付：掘金、QMT 等平台使用的信号文件、执行入口映射和运行摘要。

边界：

- L5 输入只能来自已归档或主管确认的 L4 预测表，以及已登记的策略规则。
- 策略智能体负责策略研发、规则归档和投产候选确认；交易智能体负责把已归档策略和已批准 L4 资产落地为平台可执行交易信号。
- 探索信号、重训信号、新模型信号不得覆盖投产信号。
- T+1 买入信号必须有 T 日预测分；预测表未覆盖 T 日时不得出信号。
- L5 不得修改 L1-L4 资产，不得以本地信号生成结果替代掘金验证结论。

### L6 回测、验证和审计证据层

资产范围：

- 本地粗筛和优化结果：`data_file/backtest_*.csv`、`data_file/optimizer_*.csv`、`data_file/selection_*.csv`。
- 掘金验证证据：`data_file/reports/*juejin*`、策略归档中的 `backtests/`。
- 审计报告：`data_file/reports/` 下的 raw、stock daily、factor、prediction、strategy、reproduction 类报告。

边界：

- 本地回测只允许作为粗筛，不得作为正式收益、回撤、夏普、胜率或达标依据。
- 正式回测结论必须以掘金平台结果为准，并保存日志、指标和策略脚本快照。
- 审计智能体只读审计和输出报告，不得补数、算因子、训练模型、出信号或改策略。

### L7 投产归档和发布层

资产范围：

- 投产策略归档：`strategy_library/production/<strategy_id>/`。
- 已发布策略资产：`data_file/published_strategies/`、`data_file/published_strategy_sources/`。
- MCP 发布和权限资产：`data_file/mcp_auth/`、`mcp_server/`。
- 掘金实盘/回测代码快照：`quant/掘金代码/` 或策略归档中的 `code_snapshot/`。
- 平台侧交易交付证据：模拟盘/实盘运行摘要、平台日志、信号快照与交接说明。

边界：

- 投产归档必须保存从数据、因子、标签、模型、预测、信号、交易执行到回测验证的证据链。
- 若进入掘金、QMT 等平台的模拟盘或实盘阶段，应由交易智能体维护平台交付口径和执行留痕，但不得擅自改动投产策略参数。
- 策略进入 `production/` 前必须具备 `strategy_manifest.json`、`report.md`、`trading_rules.json`、`reproduction_<version>.json`、`validation.json`、`model_params.json`、`factors.json`。
- 发布层不得重新计算底层数据资产；只能引用已验证、已归档的版本。

### 临时资产和命名规则

- 带 `smoke`、`tmp`、`repro_tmp`、`backup`、`before_*`、`*_old_*`、`*_legacy_*`、`*_probe*` 的目录或文件默认不是正式投产资产。
- 位于 `data_file/reports/` 的文件是证据，不是业务输入，除非某个脚本明确声明读取该报告作为审计门禁。
- 位于 `data_file/logs/`、`data_file/*.log`、`data_file/*.err.log` 的文件是运行日志，不是训练、预测或信号输入。
- 新增正式资产必须有明确层级、负责人、生成入口、输入依赖、审计报告和是否可投产标记。

### 跨层交接规则

```text
L1 原始数据 -> L2 综合底表：必须有 raw parquet 覆盖、raw_table_dbs/[table].DB 同步、raw audit 通过。
L2 综合底表 -> L3 因子标签：必须有 STOCK_DAILY_DATA.db::STOCK_DAILY_DATA 覆盖、复权口径确认、底表审计通过。
L3 因子标签 -> L4 模型预测：必须有投产因子表、单独标签表、未来信息隔离审计。
L4 模型预测 -> L5 策略信号：必须有预测表覆盖、模型版本说明、策略规则版本。
L5 策略信号 -> L6 回测验证：必须有信号快照、交易规则、回测区间和执行成本说明。
L6 回测验证 -> L7 投产发布：必须有掘金验证证据、策略归档和复现文件。
```

任一层未通过审计时，链路必须停在当前层，并在 `data_file/reports/` 或 `data_file/runtime/agent_memory/` 中记录原因；不得下游补救、不得跳层执行。

## 最新因子生产工作流与约束

本节为当前最高优先级的数据与因子生产规范。后续所有数据更新、因子重建、模型训练、策略调参、投产复现和掘金验证，都必须按本节区分“工作流”和“约束”执行。

### 工作流

总体链路：

```text
Tushare 原始数据
-> 原始 parquet 层
-> 原始数据审计
-> L1 raw split DB: raw_table_dbs/[table].DB
-> L2: STOCK_DAILY_DATA.db::STOCK_DAILY_DATA
-> raw_factor_data 原始因子表
-> production_factor_parts 投产特征
-> prediction_label_parts 训练标签
-> 模型训练 / L4 预测资产
-> 策略信号 CSV
-> 掘金平台回测 / 实盘脚本
```

核心原则：

```text
原始因子计算：按股票维度计算
因子标准化：按交易日维度计算
```

全量铺底用于首次建库、历史补数、复权口径变更、因子公式修复、GTJA Alpha 修复、重新训练模型、重新验证投产策略：

```text
1. 检查配置：Tushare token、数据目录、掘金策略目录、训练参数、回测日期范围。
2. 全量拉取/补齐原始数据：daily_data、daily_basic、stk_factor、adj_factor、moneyflow、limit_list_data、cyq_perf、stock_st、index_daily、daily_index_data。
3. 审计原始 parquet：文件、日期覆盖、交易日缺口、最新交易日行数、关键字段缺失率。
4. 同步 L1 raw split DB：原始 parquet 审计通过后才允许同步，默认目标为 `raw_table_dbs/[table].DB`。
5. 重建或抽取 L2 `STOCK_DAILY_DATA.db::STOCK_DAILY_DATA`：合并行情、估值、资金流、筹码、涨跌停、ST、指数、复权因子，open/high/low/close 必须统一前复权口径。
6. 审计 `STOCK_DAILY_DATA.db::STOCK_DAILY_DATA`：日期、股票、前复权、涨跌停、ST、成交额、换手率、市值字段。
7. 原始因子计算：按股票维度，计算 rolling、shift、corr、ts_rank、技术指标、资金流、筹码、GTJA 所需时间序列中间量，输出 raw_factor_data 分片。
8. 因子标准化：按交易日维度，每个 trade_date 读取当天全 A 股票 raw_factor，做缺失处理、极值处理、归一化、z-score、rank、行业中性化，生成最终 GTJA Alpha cross-sectional rank，输出 `production_factor_parts/`；训练标签输出到 `prediction_label_parts/`。
9. 因子审计：最新交易日覆盖、缺失率、极值、GTJA 全市场 rank、未来标签隔离、复权口径。
10. 模型训练：默认 expanding2010，每折训练起点固定为 20100101，每折训练终点只能到测试期开始前，并按标签周期 embargo。
11. 生成预测表：检查日期覆盖、股票覆盖、pred_prob 分布、未来信息隔离。
12. 生成策略信号：应用投产规则，过滤 ST、退市、涨停买不进、流动性条件。
13. 掘金平台回测验证：正式结论只以掘金平台结果为准，保存日志、指标、策略脚本快照。
```

每日增量用于每天收盘后更新最新数据、生成下一交易日开盘信号：

```text
1. 检查 Tushare 数据是否到齐；任一关键接口未到，必须写日志、发警告、停止出信号，不能拿旧数据冒充新信号。
2. 增量拉取最新交易日原始数据：daily_data、daily_basic、stk_factor、adj_factor、limit_list_data、cyq_perf、stock_st、index_daily；moneyflow 可按策略要求处理，但缺失必须记录。
3. 审计最新交易日原始数据：目标交易日、行数、关键字段缺失率。
4. 同步 L1 raw split DB：upsert 最新交易日及必要回看窗口，默认目标为 `raw_table_dbs/[table].DB`。
5. 更新 L2 `STOCK_DAILY_DATA.db::STOCK_DAILY_DATA`：如复权因子变化，必须回刷受影响日期窗口。
6. 原始因子增量计算：按股票维度，回看足够窗口，重算受影响股票的 raw_factor 分片。
7. 因子标准化增量计算：按交易日维度，对最新交易日读取全 A raw_factor；对受复权或原始数据变化影响的历史日期同步重算；全市场统一做 rank / z-score / 归一化 / GTJA Alpha。
8. 因子审计：最新交易日是否存在、策略所需因子是否缺失、GTJA rank 是否通过、未来标签不参与实盘预测。
9. 模型预测：生产默认使用已归档模型；若明确要求滚动重训，必须遵守 expanding2010 规则。
10. 生成 T+1 开盘信号：使用 T 日因子和预测分，输出买入、卖出、持仓处理建议。
11. 输出运行证据：signals_latest.csv、daily_signal_summary.json、数据质量报告、运行日志。
```

### 约束

维度约束：

```text
原始因子计算：按股票维度计算。
因子标准化：按交易日维度计算。
```

按股票维度计算的内容包括：rolling 均线、波动率、收益率、相关系数、shift / delay / future label 的原始辅助列、ts_rank、单股票技术指标、单股票资金流、筹码、价格衍生字段、GTJA 所需时间序列中间量。

按交易日维度计算的内容包括：全市场 rank / percentile rank、z-score 标准化、min-max 归一化、winsorize / 极值处理、行业中性化、GTJA Alpha 最终 cross-sectional rank、标签 rank。

禁止事项：

- 禁止在股票批次内计算最终全市场 rank。
- 禁止在股票批次内计算最终 GTJA Alpha cross-sectional rank。
- 禁止把 `post*`、`*_yield_rate`、`*_tag` 等未来标签列作为训练特征。
- 禁止在原始数据未审计通过时生成信号。
- 禁止在 Tushare 关键接口未到齐时拿旧数据冒充最新信号。
- 禁止把本地回测收益作为正式结论。
- 禁止未完成掘金验证就宣称策略达标或可投产。
- 禁止在存在写库/写因子冲突进程时并发重建 `raw_table_dbs/[table].DB`、`STOCK_DAILY_DATA.db`、`production_factor_parts/` 或 L4 预测资产；旧 `odb.db` 仅限 legacy/迁移场景使用。

GTJA Alpha 约束：

```text
GTJA Alpha 中的 RANK() 是全市场截面 rank，必须在因子标准化阶段按 trade_date 全 A 股票统一计算。

正确流程：
按股票计算 GTJA 时间序列中间量
-> 合并所有股票 raw_factor
-> 按 trade_date 读取全 A 截面
-> 计算 cross-sectional rank
-> 生成 gtja_alpha001 ~ gtja_alpha191
-> 运行 GTJA rank audit
```

当前已知事故：

```text
旧版 stock_factor_data.parquet 中 gtja_alpha001 ~ gtja_alpha191 存在批内 rank 风险。
审计样例：20100104 gtja_alpha101 non_null=1137, unique_count=455, unique_ratio=0.4002。
```

后续训练如果使用 GTJA Alpha，必须先确认 GTJA 分块标准化流程已经完成落盘、`gtja_alpha_rank_audit` 通过、审计报告保存到 `data_file/reports/`、训练报告中明确记录使用的是修复后的 GTJA Alpha。

复权口径约束：

- 若策略要求前复权，`STOCK_DAILY_DATA` 中 `open/high/low/close` 必须使用前复权口径。
- 技术因子应优先使用 qfq 字段。
- 不得在同一正式模型中无说明地混用 qfq、hfq、bfq。
- 如 `adj_factor` 历史发生变化，必须回刷受影响窗口或执行全量铺底。

训练与回测约束：

- 模型训练默认 `expanding2010`。
- 每折训练起点固定为 `2010-01-01`。
- 每折训练终点只能到测试期开始前。
- 必须按标签周期设置 embargo。
- 本地回测只允许作为粗筛。
- 最终收益、回撤、夏普、胜率、交易次数必须以掘金平台为准。

## 0. 祖训：模型训练窗口硬约束

做任何“正式候选策略”“投产策略”“复现投产策略”“以掘金回测为目标的调参”时，模型训练窗口必须遵守以下硬约束：

- 默认训练方式必须是 `expanding2010`。
- 每一折训练集起点固定为 `2010-01-01`，不得擅自改成 2年、3年、5年固定滚动窗口。
- 每一折训练终点只能到测试期开始前，并按标签周期保留必要 embargo，避免未来信息进入训练集。
- 如果临时探索必须使用 2年、3年、5年窗口，必须在运行前明确声明“这不是正式候选，只是探索实验”，并在报告、表名、日志中标明 `non_production_exploration`。
- 未经用户明确同意，不得把非 `expanding2010` 训练结果称为“最优版”“投产候选”“可替换生产策略”。
- 策略报告必须写清楚每一折的 `train_start`、`train_end`、`test_start`、`test_end`，并说明训练起点是否固定为 `2010-01-01`。
- 任何违反本约束得到的高收益结果，默认无效，必须重新用 `expanding2010` 全流程训练、预测，并以掘金平台回测重新验证。
- 所有正式回测结论必须以掘金平台结果为准；本地回测、本地组合调参、本地年化收益只能作为候选粗筛，不得作为最终收益、回撤、夏普、胜率或是否达标的判断依据。
- 策略报告和最终答复中必须明确区分“本地粗筛结果”和“掘金平台验证结果”。如果尚未完成掘金验证，必须标注“未完成掘金验证，不能作为正式结论”。

这条优先级高于普通调参便利性、训练速度和本地回测收益。宁可慢，也不能用错误训练窗口给出正式结论。

本文件用于说明本项目的初始化部署目标、关键配置项和自动化运行约定，方便后续由 Codex、脚本或维护人员按统一口径完成项目交付。

## 初始化部署目标

项目从 GitHub 下载后，应尽量做到开箱即用：用户只需要补充必要配置，即可完成历史数据铺底、因子计算、策略回测验证，并按照投产策略生成每日交易信号。

## 1. 环境本地部署

初始化时需要先完成本地运行环境准备，确保项目脚本、数据任务、模型训练和回测流程可以在用户机器上直接执行。

本地部署要求：

- 安装 Python 运行环境，并优先使用项目推荐版本。
- 在项目根目录创建虚拟环境，例如 `.venv/`。
- 安装 `requirements.txt` 中的依赖。
- 复制 `.env.example` 为 `.env`，复制 `config.example.json` 为 `config.json`。
- 检查本地目录是否具备读写权限，尤其是 `data_file/`、`log/`、`logs/` 和 `data_file/reports/`。
- 检查 Tushare、掘金平台、本地数据库和策略输出路径是否可用。

本地部署应尽量由初始化脚本统一完成，避免要求用户手工创建运行目录或手工修改代码路径。

## 2. 输入配置文件

初始化时需要引导用户配置以下内容：

- `TUSHARE_TOKEN`：用于从 Tushare 拉取 A 股基础行情、财务、因子所需数据。
- 掘金 / QMT 平台策略文件路径：用于连接本地生成的信号文件和掘金 / QMT 回测、模拟盘或实盘策略入口。

推荐配置方式：

- `.env`：保存本地环境变量，例如 `TUSHARE_TOKEN`。
- `config.json`：保存项目运行配置，例如数据目录、信号文件、掘金策略路径。
- `config.example.json` / `.env.example`：作为 GitHub 用户首次初始化时的模板。

不得把真实 Token、个人账号信息或本机绝对路径提交到 GitHub。

## 3. 全量补齐历史数据和因子

初始化部署需要执行一次全量数据铺底：

- 创建 `data_file/`、`log/`、`logs/`、`data_file/reports/`、`juejin_strategies/` 等运行目录。
- 拉取并补齐历史行情、指数、基础信息、财务和资金流等原始数据。
- 生成或更新 L1 raw split DB：`data_file/raw_table_dbs/[table].DB`。
- 生成或更新 L2 综合底表：`data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA`。
- 全量计算策略所需投产因子到 `data_file/production_factor_parts/`，标签单独进入 `data_file/prediction_label_parts/`。
- 生成模型训练和预测所需的 L3/L4 标准资产；L4 独立预测资产方案由模型智能体提出并经审计后接入。

历史数据和因子补齐必须支持断点续跑，避免中途失败后从头重跑。

## 4. 回测验证部署效果

数据和因子补齐完成后，必须对当前投产策略执行回测验证。

当前默认投产策略由 `strategy_library/registry.json` 中的 `production.current` 指定。现阶段默认策略为：

```text
prod_liq_prime_one_v20260612
```

策略名称：

```text
流动性精选一号
```

部署成功的判断标准：

- 能读取投产策略归档文件。
- 能读取或生成策略所需预测表。
- 能生成买入/卖出信号文件。
- 能完成本地或掘金回测。
- 回测输出指标包含累计收益、年化收益、最大回撤、夏普比率、胜率和交易次数。
- 回测结果文件写入 `data_file/reports/`。

如果回测无法执行、指标为空、信号文件为空或关键数据缺失，应视为部署未完成。

## 5. 已投产策略自动化任务

初始化成功后，应只按照策略库中已投产策略生成每日自动化任务，不允许探索策略或临时实验策略进入自动任务。

当前已投产策略清单：

| 策略 ID | 策略名称 | 任务状态 | 规则来源 |
| --- | --- | --- | --- |
| `prod_liq_prime_one_v20260612` | 流动性精选一号 | 已投产 | `strategy_library/production/prod_liq_prime_one_v20260612/` |

自动化任务目标：

- 每日 24:00 执行已投产策略自动化流程，完成数据更新、因子更新、模型预测和信号生成。
- 输出下一交易日的买入信号和卖出信号。
- 信号文件应保存到 `data_file/`，并在日志中记录生成时间、策略 ID、信号日期和买入日期。
- 自动任务必须默认读取 `strategy_library/registry.json` 中登记的已投产策略。
- 平台侧交易信号交付应由交易智能体承接，不得由未归档研究策略直接进入平台模拟盘或实盘。
- 自动任务配置保存在 `config/production_tasks.example.json`，本地使用时可复制为 `config/production_tasks.json` 后按机器环境调整。
- 自动任务入口为 `run_production_tasks.py`，Windows 定时任务安装入口为 `install_production_scheduled_task.ps1`。

建议每日任务流程：

```text
1. 检查环境和配置
2. 更新最新原始数据
3. 增量更新因子
4. 运行投产模型预测
5. 应用投产策略选股规则
6. 生成下一交易日买入信号
7. 根据持仓和策略规则生成卖出信号
8. 输出信号文件、摘要文件和日志
9. 如果任一步失败，写入错误日志并停止后续交易信号发布
```

自动化任务应保证可追踪：

- 每次运行必须写日志。
- 每次信号生成必须保留历史快照。
- 每次策略版本变更必须更新策略库归档。
- 不允许未登记的探索策略覆盖投产策略信号。

## 6. 投产策略完整复现归档规范

任何策略进入 `strategy_library/production/` 或发布新的投产版本时，必须同步建立完整复现归档。归档目标不是只保存策略结论，而是保存从底层数据、因子、标签、模型训练、预测、信号生成、交易执行到回测验证的完整证据链。

智能体在发布、修改或复测投产策略时，必须优先检查该策略目录是否具备以下文件和子目录；如果缺失，应在报告中明确标注“不可严格复现”的原因，并优先补齐。

### 6.1 必须归档的核心文件

每个投产策略目录至少应包含：

- `strategy_manifest.json`：策略 ID、名称、状态、版本、预测标签、训练口径、模型类型、规则摘要和当前信号。
- `report.md`：面向维护人员的中文说明，包含策略定义、数据集划分、因子筛选、模型参数、买卖规则和回测效果。
- `trading_rules.json`：买入、卖出、持仓、仓位、流动性过滤、涨跌停处理、滑点和执行时间等交易规则。
- `reproduction_<version>.json`：复现入口、路径、命令、环境变量、回测窗口、资金、滑点、预期指标和证据日志。
- `validation.json`：回测结果、验证时间、验证环境、核心指标和历史版本对比。
- `model_params.json`：模型类型、任务类型、完整参数、随机种子、设备、损失函数、评估函数和环境变量覆盖值。
- `factors.json`：最终投产使用的因子列表、因子数量、筛选方法和来源文件。

### 6.2 必须归档的训练与模型证据

严格复现模型预测分时，必须保存：

- 每个 fold 的因子筛选文件，例如 `selected_features_<label>_rolling_foldN.json`。
- 每个 fold 的因子 IC 明细，例如 `feature_ic_scores_<label>_rolling_foldN.csv`。
- 每个 fold 的模型文件，例如 `model_foldN.json`、`model_foldN.pkl` 或 XGBoost `booster_foldN.ubj`。
- 每个 fold 的训练窗口和测试窗口元数据，包括 `train_start`、`train_end`、`test_start`、`test_end`、样本行数和特征数。
- 训练样本、测试样本或可校验 hash；如果样本文件过大，至少保存数据表名、字段清单、行数、日期范围和内容 hash。
- 训练参数快照，必须包含 `n_estimators`、`learning_rate`、`max_depth`、`subsample`、`colsample_bytree`、`reg_alpha`、`reg_lambda`、`random_state`、`objective`、`eval_metric` 和 `device`。

如果没有模型文件或训练样本 hash，即使能重新训练，也只能称为“流程复现”，不能称为“严格复现”。

### 6.3 必须归档的预测、信号和回测证据

策略发布时必须保存：

- 最终预测表导出文件，优先使用 parquet；也可以保存 SQLite 表名、行数、日期范围和表 hash。
- 每日生产信号文件，包括买入信号、卖出信号、信号日期、执行日期和目标仓位。
- 最新信号快照，例如 `signals_latest.csv` 或策略专属 latest 文件。
- 回测日志原文，例如掘金回测 log。
- 回测指标 JSON，包含累计收益、年化收益、最大回撤、夏普比率、Calmar、胜率、开仓次数、平仓次数、初始资金和滑点。
- 如果使用外部交易或回测平台，必须保存平台策略代码快照，尤其是掘金策略目录中的 `main.py`。

### 6.4 必须归档的代码和环境快照

投产策略依赖的代码和环境必须可追踪：

- 保存信号生成脚本、回测脚本、策略执行脚本的版本或快照。
- 保存 `requirements.txt`、`conda_env.yml` 或等价依赖文件。
- 保存 Python 版本、关键三方库版本、XGBoost 版本、pandas 版本、numpy 版本。
- 保存掘金 SDK 版本、Tushare 版本和其他外部平台 SDK 版本。
- 不得把真实 Token、账号、私钥或本机个人路径写入可公开提交的归档文件。

### 6.5 推荐目录结构

投产策略建议采用以下结构：

```text
strategy_library/production/<strategy_id>/
  strategy_manifest.json
  report.md
  trading_rules.json
  reproduction_<version>.json
  validation.json
  model_params.json
  factors.json

  code_snapshot/
    main.py
    signal_generation_scripts.txt
    backtest_entrypoint.txt

  feature_selection/
    fold1_selected_features.json
    fold1_feature_ic_scores.csv
    fold2_selected_features.json
    fold2_feature_ic_scores.csv

  models/
    fold1_xgb_model.json
    fold2_xgb_model.json
    model_hashes.json

  datasets/
    fold1_train_meta.json
    fold1_test_meta.json
    dataset_hashes.json

  predictions/
    production_prediction_table.parquet
    prediction_table_meta.json

  signals/
    production_signals.csv
    signals_latest.csv

  backtests/
    gm_backtest_<version>.log
    gm_backtest_<version>_metrics.json

  environment/
    requirements.txt
    conda_env.yml
    python_version.txt
    sdk_versions.txt
```

### 6.6 智能体执行约束

智能体处理投产策略时必须遵守：

- 发布新投产版本前，先检查模型文件、训练参数、因子筛选、预测表、信号表、回测日志和环境快照是否齐全。
- 任何缺失项都必须写入策略报告和复现文件的 `caveats` 或“注意事项”中。
- 不得把一次快速复训结果描述为严格复现，除非模型文件、样本 hash、参数快照和预测结果能逐项对齐。
- 如果复训预测分与投产表不一致，应优先检查模型 checkpoint、参数环境变量、训练样本、因子文件和数据清洗口径。
- 如果只能证明训练流程一致，应明确表述为“流程复现”；只有预测分、信号和回测指标逐项一致，才可表述为“严格复现”。
- 每次投产策略版本变更后，必须更新 `strategy_manifest.json`、`trading_rules.json`、`validation.json`、`report.md` 和对应 `reproduction_<version>.json`。

## 7. 最新交易日数据链路处理规范

当用户要求补齐某个最新交易日、重算因子、生成下一交易日信号，或询问“为什么不能生成某日信号”时，智能体必须先梳理并验证完整数据链路，不能直接跳到模型预测或信号推荐。

### 7.1 数据链路顺序

本项目当前标准链路为：

```text
Tushare 原始数据
-> 原始 parquet 层
-> L1 raw split DB: raw_table_dbs/[table].DB
-> L2: STOCK_DAILY_DATA.db::STOCK_DAILY_DATA
-> L3 features: production_factor_parts/
-> L3 labels: prediction_label_parts/
-> L4 预测资产（由模型智能体提出独立方案并经审计接入）
-> 策略过滤与信号 CSV
-> 掘金回测/实盘脚本
```

其中：

- 原始 parquet 层是数据补齐的第一落点，位于 `../data_file/`。
- `raw_table_dbs/[table].DB` 是 L1 SQLite 原始表默认落点；旧 `odb.db` 只作为 legacy、回滚和迁移审计资产。
- `STOCK_DAILY_DATA.db::STOCK_DAILY_DATA` 是 L2 综合日线底表，包含前复权价格、基础行情、估值、资金流、筹码、技术指标、涨停和指数字段。
- `production_factor_parts/` 是 L3 模型训练和预测的默认特征资产。
- `prediction_label_parts/` 是 L3 训练标签资产，不得混入投产特征。
- `stock_factor_data.parquet` 是历史兼容宽表，含未来/标签列风险，不得作为新链路默认训练或预测特征输入。
- `stock_predict_data_*` 若仍位于旧 `odb.db`，只能作为 legacy 或既有投产归档复现资产；新 L4 默认预测资产需等待模型智能体方案和审计通过。
- 投产策略必须读取投产归档指定的预测表和信号规则，不能自动切换到重训表或探索表。

### 7.2 操作前必须做的审计

补数或重算前，必须先审计原始 parquet 层，至少检查以下文件：

- `daily_data.parquet`
- `daily_index_data.parquet`
- `stk_factor.parquet`
- `moneyflow.parquet`
- `limit_list_data.parquet`
- `cyq_perf.parquet`
- `adj_factor.parquet`
- `stock_st.parquet`
- `index_daily.parquet`

审计项必须包括：

- 文件是否存在。
- `min(trade_date)`、`max(trade_date)`、交易日数量。
- 最新目标交易日是否有行数。
- 最近交易日窗口是否缺日期。
- 目标交易日股票数或指数数是否明显异常。

审计结果必须写入 `../data_file/reports/`，例如：

```text
raw_data_audit_<target_date>_before_fill.csv
raw_data_audit_<target_date>_after_fill.csv
```

### 7.3 补数顺序

如果发现缺口，必须按以下顺序处理：

1. 先补历史缺失交易日，不要只补最新一天。
2. 历史缺口补齐后，再补目标最新交易日。
3. 先只更新原始 parquet，不要一边下载一边重建 SQLite 或因子。
4. 原始 parquet 全部审计通过后，再同步 L1 raw split DB：`raw_table_dbs/[table].DB`。
5. L1 raw split DB 同步完成后，再重建或增量更新 `STOCK_DAILY_DATA.db::STOCK_DAILY_DATA`。
6. `STOCK_DAILY_DATA.db::STOCK_DAILY_DATA` 覆盖目标交易日后，再计算或增量更新 `production_factor_parts/`，标签单独进入 `prediction_label_parts/`。
7. `production_factor_parts/` 和必要标签资产覆盖目标交易日后，才允许执行模型预测；预测覆盖后才允许策略信号生成。

如果任一步失败，必须停止后续步骤，并明确说明当前链路停在哪一层。

### 7.4 并发和锁库处理

补数、重建综合表、重算因子前，必须检查是否存在会读写同一数据目录或 SQLite 的后台进程：

```text
run_all_a_raw_update.py
run_cdb_update.py
run_incremental_cdb_update.py
rebuild_factor_data_batched.py
rolling_train_module.py
run_parallel_expanding2010_folds.py
run_pdb_update.py
```

如果存在旧训练、旧预测或旧补数进程，必须判断是否会锁库或写同一文件。会冲突时应先停止或等待完成，不能并发重建 `raw_table_dbs/[table].DB`、`STOCK_DAILY_DATA.db`、`production_factor_parts/`、`prediction_label_parts/` 或 L4 预测资产。旧 `odb.db` 仅在 legacy/迁移/归档复现场景下显式使用。

如果 SQLite 出现 `database is locked`，不得反复强行重跑全链路。必须先查进程、确认锁来源，再继续。

### 7.5 前复权因子计算要求

用户要求“通过前复权”或“计算前复权全部因子”时，必须确保：

- `adj_factor.parquet` 已覆盖目标交易日。
- `STOCK_DAILY_DATA.db::STOCK_DAILY_DATA` 中 `open`、`close`、`high`、`low` 使用前复权口径。
- 前复权综合表覆盖目标交易日。
- `production_factor_parts/` 重新计算或增量计算后覆盖目标交易日；如涉及训练，`prediction_label_parts/` 必须独立覆盖相应标签区间。
- 因子计算完成后必须检查目标交易日行数、核心 qfq/hfq 因子列和策略所需因子列。

不得只补 `daily_data.parquet` 就宣称“因子已更新”。

### 7.6 策略信号生成前的口径确认

生成买入/卖出信号前，必须确认：

- 策略 ID 和版本，例如 `prod_liq_prime_one_v20260612 / V1.0`。
- 使用的预测表是否为该策略归档指定表。
- 该预测表是否覆盖信号日。
- 如果原版模型 checkpoint 缺失，则不能把重训预测称为原版信号。
- 如果使用重训表生成信号，必须标记为“新模型信号”或“探索信号”，不能覆盖投产信号。

对于 `T+1` 买入信号，必须有 `T` 日预测分；例如生成 `20260616` 买入信号，必须先有 `20260615` 的预测表记录。

### 7.7 当前已知事故经验

本项目曾出现过以下问题，后续必须避免：

- 只补了 `daily_data.parquet`，但 `daily_basic`、`moneyflow`、`stk_factor`、`adj_factor` 等原始表仍滞后，导致无法生成完整因子。
- SQLite 同步时遇到 `database is locked`，导致 `STOCK_DAILY_DATA` 被删除后未成功重建。
- 当前运行的滚动训练进程继续占用 `odb.db`，影响补数和建表。
- 原版投产模型 checkpoint 未保存，因此不能用重训模型冒充原版投产信号。
- 重训模型和原版预测表在重叠区间 Top1 命中率较低，必须先做一致性验证后再讨论是否可用。

遇到类似情况时，优先恢复数据链路完整性，再讨论模型和信号；不要反过来。

## 8. Current production factor table policy (2026-06-16)

The current standard production factor wide table is:

```text
D:\work\quant\quant_mcp\quant\data_file\production_factor_parts\
```

This path is a junction to:

```text
D:\work\quant\quant_mcp\quant\data_file\production_factor_parts_clean_20260616\
```

Current production factor policy:

- Production parts are built from `STOCK_DAILY_DATA`-driven raw factor parts, not from auxiliary table date sets.
- Legacy-only GTJA smoke assets may use `raw_gtja_identity`, but the maintained L3 default uses official full-market `RANK(x)` cross-sectional rank by `trade_date`.
- Default production schema excludes future/label columns.
- Default production schema excludes source-limited full-history fields from `cyq_perf`, `stock_st`, and `limit_list_data`.
- Default production schema excludes non-default auxiliary features from `top_list`, `ths_hot`, and `dc_hot` sources, including `ths_hot`, `ths_rank`, and `dc_rank`.
- `adj_factor` non-trading source dates `20210920` and `20210921` must not expand production factor dates.
- Default production schema includes the current market/price auxiliary fields
  `open`, `high`, `low`, `close`, `pre_close`, `amount`, and `vol`.
- Default production schema may include `industry_encode` only when it is
  derived from the governed mapping asset
  `data_file/runtime/production_factor_industry_encode_mapping.json`; the
  legacy `stock_factor_data.parquet.industry_encode` field is not an approved
  default source for stable industry encoding.

The target-date incremental factor update entrypoint is:

```text
python quant\main\incremental_factor_update_target_date.py --target-date <YYYYMMDD>
```

For `20260616`, final audit evidence is:

```text
D:\work\quant\quant_mcp\quant\data_file\reports\incremental_factor_update_20260616_final_audit.json
```

Audit result for `20260616`: raw and production factor parts both have 5513 rows / 5513 distinct stocks for the target date; the 2026-06-17 clean production schema had 833 columns with no future columns, no source-limited columns, and no non-default auxiliary columns. As of 2026-06-18, the current production schema adds governed market/price auxiliary fields plus governed `industry_encode`, bringing the current production schema to 841 columns.
