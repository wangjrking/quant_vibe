param(
    [string]$TaskName = "QuantVibeProductionStrategies",
    [string]$Python = "python",
    [string]$ProjectDir = $PSScriptRoot,
    [string]$Config = (Join-Path $PSScriptRoot "config\production_tasks.example.json"),
    [string]$RunAt = "00:00"
)

$ErrorActionPreference = "Stop"

$runner = Join-Path $ProjectDir "run_production_tasks.py"
if (-not (Test-Path $runner)) {
    throw "Runner not found: $runner"
}
if (-not (Test-Path $Config)) {
    throw "Config not found: $Config"
}

$arguments = "`"$runner`" --config `"$Config`""
$action = New-ScheduledTaskAction -Execute $Python -Argument $arguments -WorkingDirectory $ProjectDir
$trigger = New-ScheduledTaskTrigger -Daily -At $RunAt
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Run Quant Vibe registered production strategy automation tasks daily at 24:00." `
    -Force | Out-Null

Write-Host "Scheduled task installed: $TaskName"
Write-Host "Run time: $RunAt daily"
Write-Host "Runner: $runner"
Write-Host "Config: $Config"
