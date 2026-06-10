# Trading Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage protection, practical backtesting, and daily candidate selection around the existing prediction table.

**Architecture:** Keep the existing data and model pipeline intact while adding small pure-Python modules at the repository root. The AI module imports only the leakage guard so tests do not need heavy ML dependencies.

**Tech Stack:** Python standard library, SQLite, existing pandas/XGBoost pipeline.

---

### Task 1: Leakage Guard

**Files:**
- Create: `leakage_guard.py`
- Test: `tests/test_leakage_guard.py`
- Modify: `ai_module.py`

- [ ] Write tests that require future-looking features to be rejected.
- [ ] Implement `find_leaky_features()` and `validate_no_leakage()`.
- [ ] Call `validate_no_leakage(train_x.columns, label=label)` before model fitting.

### Task 2: Practical Backtest

**Files:**
- Create: `backtest_module.py`
- Test: `tests/test_backtest_module.py`

- [ ] Write tests for limit-up buy blocking and transaction-cost-adjusted returns.
- [ ] Implement row filtering, daily top-k selection, return calculation, equity curve, drawdown, and RankIC.
- [ ] Add a CLI that can read `../data_file/odb.db` and print metrics.

### Task 3: Daily Selection

**Files:**
- Create: `selection_module.py`
- Test: `tests/test_selection_module.py`

- [ ] Write tests for latest-date selection, ST filtering, ATR filtering, and per-industry caps.
- [ ] Implement `select_candidates()` and a CLI that writes a CSV candidate file.

### Task 4: Runtime Fixes

**Files:**
- Modify: `data_load_module.py`
- Modify: `message_module.py`

- [ ] Replace undefined `store_` with `store_daily_index_data`.
- [ ] Guard email sending with `if __name__ == "__main__":`.

### Task 5: Verification

**Commands:**
- `python -m unittest discover -s tests -v`
- `python -m py_compile leakage_guard.py backtest_module.py selection_module.py ai_module.py data_load_module.py message_module.py`
- `python backtest_module.py --db ..\data_file\odb.db --table stock_predict_data_10d_yield_rate --start 20260105 --top-k 10`
- `python selection_module.py --db ..\data_file\odb.db --table stock_predict_data_10d_yield_rate --top-k 10`
