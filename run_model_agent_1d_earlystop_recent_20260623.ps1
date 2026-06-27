$ErrorActionPreference = "Stop"

Remove-Item Env:\XGB_SAMPLE_WEIGHT_MODE -ErrorAction SilentlyContinue
Remove-Item Env:\XGB_SAMPLE_WEIGHT_TOP_PCT -ErrorAction SilentlyContinue
Remove-Item Env:\XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER -ErrorAction SilentlyContinue

$env:XGB_REG_EVAL_METRIC = "top_return_loss"
$env:XGB_TOP_RETURN_EVAL_K = "10"
$env:XGB_VALIDATION_MODE = "train_tail_days"
$env:XGB_VALIDATION_TAIL_DAYS = "126"

$python = "D:\work\quant\quant_mcp\.venv\Scripts\python.exe"
$root = "D:\work\quant\quant_mcp"
$main = Join-Path $root "quant\main"
$out = "../data_file/reports/model_agent_research_1d_earlystop_recent_20260623"
$selected = "../data_file/reports/model_agent_research_1d_next_20260622_smoke/feature_scores/xgb_reg1d_research_next_20260622_smoke/selected_features_executable_1d_open_return_rolling_fold1.json"

Set-Location $main

& $python run_parallel_expanding2010_folds.py `
  --data-file-url ../data_file `
  --data-start 20100101 `
  --first-test 20240604 `
  --final-test 20260622 `
  --label executable_1d_open_return `
  --model-type reg `
  --train-years 5 `
  --test-months 3 `
  --step-months 3 `
  --embargo-days 10 `
  --selected-features-path $selected `
  --output-table stock_predict_data_model_agent_research_1d_earlystop_recent_20260623 `
  --output-dir $out `
  --experiment-name xgb_reg1d_earlystop_recent_20260623 `
  --start-fold 8 `
  --end-fold 9 `
  --max-workers 1 `
  --xgb-device cuda `
  --xgb-n-jobs 0 `
  --xgb-n-estimators 5000 `
  --xgb-learning-rate 0.02 `
  --xgb-max-depth 2 `
  --xgb-reg-lambda 10 `
  --xgb-reg-alpha 1 `
  --xgb-early-stopping-rounds 200 `
  --prediction-output-mode independent `
  --skip-existing
