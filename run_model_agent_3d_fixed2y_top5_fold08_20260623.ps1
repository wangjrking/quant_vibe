$ErrorActionPreference = "Stop"

$env:XGB_REG_EVAL_METRIC = "top_return_loss"
$env:XGB_TOP_RETURN_EVAL_K = "5"
$env:XGB_VALIDATION_MODE = "train_tail_days"
$env:XGB_VALIDATION_TAIL_DAYS = "126"
$env:XGB_SAMPLE_WEIGHT_MODE = "daily_top_quantile"
$env:XGB_SAMPLE_WEIGHT_TOP_PCT = "0.005"
$env:XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER = "10.0"
$env:XGB_DEVICE = "cuda"
$env:XGB_N_JOBS = "0"
$env:XGB_N_ESTIMATORS = "3000"
$env:XGB_LEARNING_RATE = "0.006"
$env:XGB_MAX_DEPTH = "2"
$env:XGB_REG_LAMBDA = "12"
$env:XGB_REG_ALPHA = "0.2"
$env:XGB_EARLY_STOPPING_ROUNDS = "300"

$python = "D:\work\quant\quant_mcp\.venv\Scripts\python.exe"
$root = "D:\work\quant\quant_mcp"
$main = Join-Path $root "quant\main"
$out = "../data_file/reports/model_agent_research_3d_fixed2y_top5_fold08_20260623"
$selected = "../data_file/reports/model_agent_research_3d_fixed2y_fold08_20260623/feature_scores/selected_features_executable_3d_open_return_rolling_fold8.json"

Set-Location $main

& $python rolling_train_module.py `
  --data-file-url ../data_file `
  --data-start 20100101 `
  --first-test 20240604 `
  --final-test 20260622 `
  --label executable_3d_open_return `
  --model-type reg `
  --train-years 2 `
  --test-months 3 `
  --step-months 3 `
  --embargo-days 10 `
  --train-mode fixed `
  --selected-features-path $selected `
  --output-table stock_predict_data_model_agent_research_3d_fixed2y_top5_fold08_20260623 `
  --prediction-output-path "$out/fold_predictions/fold08.parquet" `
  --summary-output "$out/fold_results.csv" `
  --start-fold 8 `
  --end-fold 8 `
  --execute
