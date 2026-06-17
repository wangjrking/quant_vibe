---
name: quant-vibe-deploy
description: Use when a cloned Quant Vibe repository needs first-time deployment, environment initialization, data download, factor rebuild, model training, prediction asset setup, strategy signal generation, or reproducibility checks.
---

# Quant Vibe Deploy

## Overview

Initialize a fresh Quant Vibe clone into a working local research environment. Keep secrets local, use the repository's layered data chain, and prefer the bundled bootstrap script for repeatable setup.

## Quick Start

From the repository root, run a dry run first:

```powershell
python .claude\skills\quant-vibe-deploy\scripts\bootstrap_quant_vibe.py --dry-run --skip-pip
```

Then require the user's private data token before real data work:

```powershell
$env:TUSHARE_TOKEN = "<user token>"
python .claude\skills\quant-vibe-deploy\scripts\bootstrap_quant_vibe.py --check --require-token
python .claude\skills\quant-vibe-deploy\scripts\bootstrap_quant_vibe.py --end-date YYYYMMDD
```

Use the project interpreter if it already exists:

```powershell
D:\work\quant\quant_mcp\.venv\Scripts\python.exe .claude\skills\quant-vibe-deploy\scripts\bootstrap_quant_vibe.py --dry-run --skip-pip
```

## Deployment Workflow

1. Read `AGENTS.md` and `README.md` before executing data, model, or strategy work.
2. Keep private values in environment variables. `TUSHARE_TOKEN` must not be written to `config.json`, logs, strategy archives, or git.
3. Run `bootstrap_quant_vibe.py --check --require-token` to verify Python, project paths, shareable config templates, and the private token.
4. Run the bootstrap script without `--dry-run` only after the user confirms the target `--end-date` and expected runtime.
5. Follow the current layered chain:
   - L1 raw data: `run_all_a_raw_update.py`
   - L3 production features: `production_factor_parts`
   - L3 labels: `prediction_label_parts`
   - L4 predictions and model evidence: `run_pdb_update.py` writes to `model_predictions`
   - L5 production task/signal entry: `run_production_tasks.py`
6. If any step fails, stop at that layer and report the failed command, log path, and next recovery action. Do not skip ahead to model or strategy generation.

## Bootstrap Script

Use `scripts/bootstrap_quant_vibe.py` for deterministic setup. Important modes:

| Mode | Use |
| --- | --- |
| `--dry-run` | Print the full install, data, factor, model, and strategy commands without executing them. |
| `--check` | Validate local prerequisites and token state without running the full chain. |
| `--skip-pip` | Do not install `requirements.txt`; useful inside an already prepared Codex environment. |
| `--require-token` | Fail if `TUSHARE_TOKEN` is missing. Use before real downloads. |
| `--limit-stocks N` | Smoke-test the raw data step on a small stock subset. |

The script is intentionally conservative: it copies `config.example.json` to `config.json` only when missing, creates runtime directories, and emits every command it plans to run.

## Reproduction Boundaries

- Current production factor input is `production_factor_parts`, not legacy `stock_factor_data.parquet`.
- Current training labels live in `prediction_label_parts` and must stay separate from production features.
- Current prediction assets should be written under `model_predictions` or a strategy archive, not to legacy `odb.db` by default.
- `run_production_tasks.py` requires an approved non-legacy prediction manifest for L5 production reads; if the clone has only placeholders, keep L5 disabled and report that model approval is still required.
- Large `data_file/`, logs, caches, local GM strategies, and tokens are local runtime assets and must remain out of git.

## Expected User Prompt

When the user says `@quant-vibe-deploy start deployment`, perform this sequence:

1. Confirm the repository root and Python interpreter.
2. Run:

```powershell
python .claude\skills\quant-vibe-deploy\scripts\bootstrap_quant_vibe.py --dry-run --skip-pip
```

3. If the plan looks right, ensure `TUSHARE_TOKEN` is present.
4. Run the real bootstrap with an explicit `--end-date`.
5. Run focused tests after setup:

```powershell
python -m pytest tests/test_project_deploy_skill.py -q
python run_production_tasks.py --config config/production_tasks.example.json --dry-run
```
