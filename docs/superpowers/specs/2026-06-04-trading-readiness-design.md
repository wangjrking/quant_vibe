# Trading Readiness Design

## Goal

Improve the project from a research-style stock predictor into a safer daily stock selection workflow that can be evaluated with executable trading assumptions before any real-money use.

## Scope

This first version does not promise profitable trading and does not retrain the model. It adds guardrails and evaluation around the existing prediction table:

- reject future-looking model features before training;
- evaluate predictions with a practical next-open-to-next-open backtest;
- generate a daily candidate list with liquidity and risk filters;
- fix obvious runtime hazards that stop repeatable operation.

## Components

### Leakage Guard

`leakage_guard.py` owns feature-name validation. It rejects fields such as `post_*`, forward return labels, future return ranks, and tags. The AI module calls this guard just before fitting.

### Backtest Module

`backtest_module.py` reads rows from `stock_predict_data_10d_yield_rate` or accepts rows in memory. It ranks each trade date by `pred_prob`, applies buyability checks, models transaction costs and slippage, then reports cumulative return, average daily return, win rate, max drawdown, average holdings, and RankIC.

### Selection Module

`selection_module.py` reads the latest prediction date and emits a practical candidate list. It excludes ST names, current limit-up rows, high ATR names, low predictions, and caps candidates per industry.

### Engineering Fixes

`data_load_module.py` should call `store_daily_index_data`, not an undefined `store_`. `message_module.py` should not send email on import.

## Testing

Unit tests use Python `unittest` and pure-Python data rows so they do not require the large local parquet files. Tests cover leakage detection, cost-adjusted returns, limit-up buy blocking, industry caps, and latest-date selection.
