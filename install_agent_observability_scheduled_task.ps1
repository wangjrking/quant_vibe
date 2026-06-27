param(
    [string]$TaskName = "QuantVibeAgentObservability",
    [string]$Python = "python",
    [string]$ProjectDir = $PSScriptRoot,
    [string]$RunAt = "00:20",
    [string]$DataDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "data_file")
)

$ErrorActionPreference = "Stop"

$runner = Join-Path $ProjectDir "run_agent_observability_daily.py"
if (-not (Test-Path $runner)) {
    throw "Runner not found: $runner"
}

$arguments = "`"$runner`" --data-dir `"$DataDir`""
$action = New-ScheduledTaskAction -Execute $Python -Argument $arguments -WorkingDirectory $ProjectDir
$trigger = New-ScheduledTaskTrigger -Daily -At $RunAt
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Run daily agent token and business-output observability snapshot." `
    -Force | Out-Null

Write-Host "Scheduled task installed: $TaskName"
Write-Host "Run time: $RunAt daily"
Write-Host "Runner: $runner"
Write-Host "Data dir: $DataDir"
