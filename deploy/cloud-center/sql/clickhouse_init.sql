CREATE DATABASE IF NOT EXISTS l1_raw;
CREATE DATABASE IF NOT EXISTS l2_base;
CREATE DATABASE IF NOT EXISTS l3_feature;
CREATE DATABASE IF NOT EXISTS l4_model;
CREATE DATABASE IF NOT EXISTS l5_strategy;
CREATE DATABASE IF NOT EXISTS l6_backtest;
CREATE DATABASE IF NOT EXISTS l7_trading;

CREATE TABLE IF NOT EXISTS l2_base.stock_daily_data
(
    trade_date Date,
    ts_code String,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    vol Float64,
    amount Float64,
    adj_factor Nullable(Float64),
    st_type Nullable(String),
    asset_id String,
    track LowCardinality(String),
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, ts_code);

CREATE TABLE IF NOT EXISTS l3_feature.factor_values
(
    trade_date Date,
    ts_code String,
    factor_name LowCardinality(String),
    factor_value Nullable(Float64),
    asset_id String,
    track LowCardinality(String),
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, factor_name, ts_code);

CREATE TABLE IF NOT EXISTS l3_feature.prediction_labels
(
    trade_date Date,
    ts_code String,
    horizon LowCardinality(String),
    label_value Nullable(Float64),
    asset_id String,
    track LowCardinality(String),
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, horizon, ts_code);

CREATE TABLE IF NOT EXISTS l4_model.predictions
(
    trade_date Date,
    ts_code String,
    horizon LowCardinality(String),
    score Float64,
    rank UInt32,
    model_id String,
    asset_id String,
    manifest_id String,
    track LowCardinality(String),
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, horizon, score, ts_code);

CREATE TABLE IF NOT EXISTS l5_strategy.strategy_signals
(
    signal_date Date,
    trade_date Date,
    ts_code String,
    strategy_id String,
    side LowCardinality(String),
    target_weight Float64,
    score Nullable(Float64),
    asset_id String,
    manifest_id String,
    track LowCardinality(String),
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(signal_date)
ORDER BY (signal_date, strategy_id, ts_code);

CREATE TABLE IF NOT EXISTS l6_backtest.backtest_results
(
    run_id String,
    strategy_id String,
    trade_date Date,
    metric_name LowCardinality(String),
    metric_value Float64,
    asset_id String,
    track LowCardinality(String),
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_date)
ORDER BY (strategy_id, run_id, trade_date, metric_name);

CREATE TABLE IF NOT EXISTS l7_trading.execution_records
(
    execution_date Date,
    ts_code String,
    strategy_id String,
    platform LowCardinality(String),
    side LowCardinality(String),
    quantity Float64,
    price Nullable(Float64),
    status LowCardinality(String),
    asset_id String,
    track LowCardinality(String),
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(execution_date)
ORDER BY (execution_date, strategy_id, platform, ts_code);
