$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\work\quant\quant_mcp"
$MainDir = Join-Path $ProjectRoot "quant\main"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$OutputDir = Join-Path $ProjectRoot "quant\data_file\reports\model_agent_research_1d_next_20260622_run2"
$LogPath = Join-Path $OutputDir "run_parallel_combined.log"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Set-Location $MainDir

& $Python run_parallel_expanding2010_folds.py `
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
  --selected-features-path ../data_file/reports/model_agent_research_1d_next_20260622_smoke/feature_scores/xgb_reg1d_research_next_20260622_smoke/selected_features_executable_1d_open_return_rolling_fold1.json `
  --output-table stock_predict_data_model_agent_research_1d_next_20260622_run2 `
  --output-dir ../data_file/reports/model_agent_research_1d_next_20260622_run2 `
  --experiment-name xgb_reg1d_research_next_20260622_run2 `
  --max-workers 1 `
  --xgb-device cuda `
  --xgb-n-jobs 0 `
  --xgb-n-estimators 3000 `
  --xgb-learning-rate 0.03 `
  --xgb-max-depth 2 `
  --xgb-reg-lambda 8 `
  --xgb-reg-alpha 1 `
  --prediction-output-mode independent `
  --skip-existing *>&1 | Tee-Object -FilePath $LogPath

exit $LASTEXITCODE
