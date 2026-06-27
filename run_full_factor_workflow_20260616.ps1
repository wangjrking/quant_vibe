$ErrorActionPreference = "Stop"

$Python = "D:\work\anaconda\python.exe"
$WorkDir = "D:\work\quant\quant_mcp\quant\main"
$DataDir = "D:\work\quant\quant_mcp\quant\data_file"
$RawDir = Join-Path $DataDir "raw_factor_by_stock_parts"
$StandardDir = Join-Path $DataDir "standard_factor_by_date_parts"
$FactorPath = Join-Path $DataDir "stock_factor_data.parquet"
$WorkflowLog = Join-Path $DataDir "full_factor_workflow_20260616.log"

function Write-WorkflowLog($Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    $line | Tee-Object -FilePath $WorkflowLog -Append
}

function Get-RawPartCount {
    if (-not (Test-Path $RawDir)) {
        return 0
    }
    return (Get-ChildItem -LiteralPath $RawDir -Filter "raw_part_*.parquet" -ErrorAction SilentlyContinue | Measure-Object).Count
}

function Get-RawWorkerCount {
    return (Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { $_.CommandLine -like '*rebuild_raw_factor_by_stock.py*' } |
        Measure-Object).Count
}

Set-Location $WorkDir
Write-WorkflowLog "workflow_start raw_dir=$RawDir standard_dir=$StandardDir"

while ($true) {
    $count = Get-RawPartCount
    $workers = Get-RawWorkerCount
    Write-WorkflowLog "raw_progress parts=$count/1106 workers=$workers"
    if ($count -ge 1106 -and $workers -eq 0) {
        break
    }
    if ($workers -eq 0 -and $count -lt 1106) {
        throw "raw workers stopped before all parts were produced: $count/1106"
    }
    Start-Sleep -Seconds 300
}

New-Item -ItemType Directory -Force -Path $StandardDir | Out-Null
$chunkLines = & $Python "rebuild_factor_data_date_chunks.py" `
    --factor-path $FactorPath `
    --chunk-days 20 `
    --past-overlap-days 25 `
    --future-overlap-days 0
$chunkArray = @($chunkLines)
$chunkCount = [int]$chunkArray.Count
Write-WorkflowLog "standard_chunks count=$chunkCount"

$lastChunk = $chunkCount - 1
$mid = [int][math]::Floor($lastChunk / 2)
$secondStart = $mid + 1
$ranges = @(
    @(0, $mid),
    @($secondStart, $lastChunk)
)

$procs = @()
for ($i = 0; $i -lt $ranges.Count; $i++) {
    $start = [int]$ranges[$i][0]
    $end = [int]$ranges[$i][1]
    if ($start -gt $end) {
        continue
    }
    $log = Join-Path $DataDir "standard_factor_worker_${i}_20260616.log"
    $err = Join-Path $DataDir "standard_factor_worker_${i}_20260616.err.log"
    $args = @(
        "rebuild_factor_data_date_chunks.py",
        "--factor-path", $FactorPath,
        "--raw-parts-dir", $RawDir,
        "--output-dir", $StandardDir,
        "--chunk-days", "20",
        "--past-overlap-days", "25",
        "--future-overlap-days", "0",
        "--start-chunk", [string]$start,
        "--end-chunk", [string]$end,
        "--rank-columns", "gtja_alpha101,gtja_alpha102,gtja_alpha103,gtja_alpha104,gtja_alpha105",
        "--add-rank",
        "--add-zscore"
    )
    Write-WorkflowLog "standard_worker_start index=$i chunk=$start..$end"
    $procs += Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $WorkDir -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
}

foreach ($proc in $procs) {
    Wait-Process -Id $proc.Id
    Write-WorkflowLog "standard_worker_done pid=$($proc.Id) exit=$($proc.ExitCode)"
    if ($proc.ExitCode -ne 0) {
        throw "standard worker failed pid=$($proc.Id) exit=$($proc.ExitCode)"
    }
}

Write-WorkflowLog "workflow_done"
