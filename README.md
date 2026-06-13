# Quant Vibe Multi-Factor Quant Project

Quant Vibe is an A-share multi-factor research and signal-generation project. It includes raw data updates, factor processing, model training, stock selection, backtesting, GM signal export, and report generation.

The project uses paths relative to the repository root by default, so a fresh clone does not need machine-specific path edits.

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
docker run --rm -e TUSHARE_TOKEN=your_token_here -v "${PWD}/data_file:/app/data_file" quant-vibe python run_cdb_update.py
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

## Common Commands

```bash
# Main flow: raw data, factor processing, model training
python main.py

# Update factor data
python run_cdb_update.py

# Train model and write predictions
python run_pdb_update.py

# Daily incremental flow on Windows PowerShell
.\run_daily_strategy.ps1

# Run registered production strategy automation tasks
python run_production_tasks.py --config config/production_tasks.example.json

# Install Windows scheduled task for daily 24:00 production strategy automation
.\install_production_scheduled_task.ps1

# Export stock pool
python export_stock_pool.py

# Generate GM strategy reports
python generate_juejin_strategy_reports.py
```

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
- Large data, logs, caches, and local GM strategies are excluded from Git and Docker build context.

This project is for quantitative research and strategy validation only. It is not investment advice. Before live trading, verify data quality, reproduce backtests, evaluate costs, configure risk controls, and test with small capital.
