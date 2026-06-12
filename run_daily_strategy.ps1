param(
    [string]$Python = "python",
    [string]$ProjectDir = $PSScriptRoot,
    [string]$DataDir = (Join-Path $PSScriptRoot "data_file"),
    [string]$Label = "10d_yield_rate",
    [string]$SourceCommand = "",
    [switch]$ForcePrediction
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "daily_strategy.py",
    "--project-dir", $ProjectDir,
    "--data-dir", $DataDir,
    "--python", $Python,
    "--label", $Label,
    "--summary-output", (Join-Path $DataDir "daily_strategy_summary.json")
)

if ($SourceCommand) {
    $argsList += @("--source-command", $SourceCommand)
}

if ($ForcePrediction) {
    $argsList += "--force-prediction"
}

Push-Location $ProjectDir
try {
    & $Python @argsList
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
