# Quant Vibe Multi-Factor Quant Project

Quant Vibe is an A-share multi-factor research and signal-generation project. It includes raw data updates, factor processing, model training, stock selection, backtesting, GM signal export, and report generation.

The project uses paths relative to the repository root by default, so a fresh clone does not need machine-specific path edits.

## Standard Chain Default

Current model-side standard chain defaults are:

- L3 production features: `quant/data_file/production_factor_parts/`
- L3 training labels: `quant/data_file/prediction_label_parts/`
- L4 prediction assets: `quant/data_file/model_predictions/`

Legacy mixed assets such as `quant/data_file/stock_factor_data.parquet`,
`quant/data_file/standard_factor_by_date_parts/`,
`quant/data_file/standard_factor_by_date_parts_fast40/`, and
`quant/data_file/odb.db::stock_predict_data_*` are archived and must not be
used as default training or prediction inputs. Model-side legacy usage now
requires explicit `QUANT_ALLOW_LEGACY_MODEL_ASSET_CHAIN=1`.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/wangjrking/quant_vibe.git
cd quant_vibe
```

### 2. Local Python Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Configure Token

Prefer environment variables so private tokens are not committed:

```powershell
$env:TUSHARE_TOKEN="your_token_here"
```

You can also copy the example config and edit it:

```bash
copy config.example.json config.json
```

Default paths:

| Item | Default path |
| --- | --- |
| Config | `config.json` |
| Data | `data_file/` |
| Logs | `log/`, `logs/` |
| Reports | `data_file/reports/` |
| GM strategies | `juejin_strategies/` |

Override the data directory when needed:

```powershell
$env:QUANT_DATA_DIR="my_data"
```

## Docker

Build the default image:

```bash
docker build -t quant-vibe .
```

Run the main pipeline with local data/log folders mounted:

```powershell
docker run --rm `
  -e TUSHARE_TOKEN=your_token_here `
  -v "${PWD}/data_file:/app/data_file" `
  -v "${PWD}/log:/app/log" `
  -v "${PWD}/logs:/app/logs" `
  quant-vibe
```

Run another script in the same image:

```powershell
docker run --rm -e TUSHARE_TOKEN=your_token_here -v "${PWD}/data_file:/app/data_file" quant-vibe python run_production_tasks.py --config config/production_tasks.example.json --dry-run
```

Use Docker Compose:

```bash
copy .env.example .env
docker compose up --build
```

Optional build args:

```bash
# Install CPU-only PyTorch for MLP/deep-learning scripts
docker build --build-arg INSTALL_TORCH=true -t quant-vibe:torch .

# Install the GM SDK if your environment can access the package
docker build --build-arg INSTALL_GM=true -t quant-vibe:gm .

# Use a custom PyPI mirror
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t quant-vibe .
```

The image stores runtime data in `/app/data_file`, logs in `/app/log` and `/app/logs`, and local GM strategies in `/app/juejin_strategies`. These directories are declared as volumes and are ignored by Git.

## Current Workflow

The current standard layered workflow is documented in:

- `WORKFLOW.md`
- `quant/main/AGENTS.md`

Current default assets:

- L1 raw split DB: `quant/data_file/raw_table_dbs/[table].DB`
- L2 market base: `quant/data_file/STOCK_DAILY_DATA.db::STOCK_DAILY_DATA`
- L3 production features: `quant/data_file/production_factor_parts/`
- L3 training labels: `quant/data_file/prediction_label_parts/`
- L4 prediction assets: independent prediction-asset plan; `odb.db.stock_predict_data_*` is not the default new-chain entry
- L5 standard automation entry: `quant/main/run_production_tasks.py`

Formal manifest contract for L5 reads:

- Production L5 reads require a prediction manifest with `approval_status=approved_for_l5`.
- The manifest must explicitly define `db_path` and `table` for a non-legacy L4 prediction asset.
- A manifest that still points to `odb.db` or a legacy prediction table is for legacy reproduction only, not the default production signal path.

## Common Commands

```bash
# Legacy historical mixed-db pipeline only
# Requires explicit opt-in:
#   QUANT_ALLOW_LEGACY_MAIN=1
python main.py

# Legacy wide-table factor rebuild only
python run_cdb_update.py

# Legacy wide-table incremental factor maintenance only
python run_incremental_cdb_update.py

# New-chain production factor parts incremental update
python incremental_factor_update_target_date.py --target-date YYYYMMDD

# 长时间挂起时的分段补跑入口：
# 对同一目标日期按 part 区间分段执行，并分别落盘 chunk 报告
python run_incremental_factor_update_chunked.py --target-date YYYYMMDD

# Train model and write predictions
python run_pdb_update.py

# Legacy mixed-asset daily orchestration only
# Requires explicit opt-in:
#   QUANT_ALLOW_LEGACY_DAILY_STRATEGY=1
.\run_daily_strategy.ps1 -AllowLegacyAssetChain

# Recommended L5 automation entry
python run_production_tasks.py --config config/production_tasks.example.json --dry-run

# Install Windows scheduled task for daily 24:00 production strategy automation
.\install_production_scheduled_task.ps1

# Export stock pool
python export_stock_pool.py

# Generate GM strategy reports
python generate_juejin_strategy_reports.py
```

`python main.py` is a legacy historical mixed-db pipeline entry retained for rollback or reproduction. It is not the recommended default entrypoint for the current layered workflow.

`daily_strategy.py` and `run_daily_strategy.ps1` are also legacy mixed-asset orchestration entries. They still depend on `stock_factor_data.parquet` and `odb.db.stock_predict_data_*`, so they require explicit opt-in and are not the standard L5 signal path.

`run_cdb_update.py` and `run_incremental_cdb_update.py` maintain the historical compatibility wide-factor parquet. The current layered default L3 feature asset remains `production_factor_parts/`.

`config/production_tasks.example.json` now ships disabled by default and should only be enabled after the model agent provides an `approved_for_l5` manifest with explicit `db_path` and `table`.

For the same reason, `config/prediction_manifests/prod_liq_prime_one_v20260612_l4_formal.json` is only a placeholder until it is updated to an actual approved L4 asset manifest. Before that, L5 automation must stay disabled.

GM-related scripts read relative paths under `juejin_strategies/` by default. Put your own GM strategy `main.py` there, or pass a strategy directory via command-line arguments where supported.

## Project Layout

| Path | Purpose |
| --- | --- |
| `project_paths.py` | Central path/config resolver |
| `config.json` | Local default config; token can be provided by env var |
| `config.example.json` | Shareable config template |
| `Dockerfile` | Reproducible runtime image |
| `docker-compose.yml` | Local container runner with mounted data/log folders |
| `.dockerignore` | Keeps data/cache/git files out of Docker build context |
| `config/production_tasks.example.json` | Example automation config for registered production strategies |
| `run_production_tasks.py` | Runs automation tasks for strategies registered as production |
| `install_production_scheduled_task.ps1` | Installs a Windows scheduled task for daily 24:00 production automation |
| `data_file/` | Local data, SQLite DB, signals, and reports; not committed |
| `log/`, `logs/` | Runtime logs; not committed |
| `juejin_strategies/` | Local GM strategies; not committed |
| `tests/` | Unit tests |

## Notes

- The repository does not depend on fixed local absolute paths.
- Default outputs are written under `data_file/`.
- `datasource.tushare_token` may stay empty in `config.json`; `TUSHARE_TOKEN` takes precedence at runtime.
- `config.json` is intentionally local-only and ignored by Git; start from `config.example.json`.
- Large data, logs, caches, and local GM strategies are excluded from Git and Docker build context.

This project is for quantitative research and strategy validation only. It is not investment advice. Before live trading, verify data quality, reproduce backtests, evaluate costs, configure risk controls, and test with small capital.
