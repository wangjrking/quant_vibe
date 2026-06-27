$ErrorActionPreference = "Stop"

$Root = if ($env:QUANT_CLOUD_CENTER_ROOT) { $env:QUANT_CLOUD_CENTER_ROOT } else { "D:\quant\cloud-center" }
$Repo = Join-Path $Root "repo"
$EnvFile = Join-Path $Root "env\warehouse.env"
$ComposeFile = Join-Path $Repo "deploy\cloud-center\docker-compose.win10.yml"
$ClickHouseUrl = if ($env:CLICKHOUSE_URL) { $env:CLICKHOUSE_URL } else { "http://127.0.0.1:8123" }
$MinioHealthUrl = if ($env:MINIO_HEALTH_URL) { $env:MINIO_HEALTH_URL } else { "http://127.0.0.1:9100/minio/health/live" }
$AssetRegistryMcpUrl = if ($env:ASSET_REGISTRY_MCP_URL) { $env:ASSET_REGISTRY_MCP_URL } else { "http://127.0.0.1:8101" }
$OpsMcpUrl = if ($env:OPS_MCP_URL) { $env:OPS_MCP_URL } else { "http://127.0.0.1:8102" }
$AuditMcpUrl = if ($env:AUDIT_MCP_URL) { $env:AUDIT_MCP_URL } else { "http://127.0.0.1:8103" }
$PostgresContainer = if ($env:POSTGRES_CONTAINER) { $env:POSTGRES_CONTAINER } else { "quant-postgres" }
$PostgresUser = if ($env:POSTGRES_USER) { $env:POSTGRES_USER } else { "quant_admin" }
$PostgresDb = if ($env:POSTGRES_DB) { $env:POSTGRES_DB } else { "quant_warehouse" }

Write-Host "[0/5] Checking Docker Desktop context..."
docker context use desktop-linux | Out-Host
docker version | Out-Host

Write-Host "[1/5] Checking Docker Compose services..."
docker compose --env-file $EnvFile -f $ComposeFile ps | Out-Host

Write-Host "[2/5] Checking ClickHouse SELECT 1..."
$clickhouseResult = Invoke-WebRequest -UseBasicParsing -Uri "$ClickHouseUrl/?query=SELECT%201"
if ($clickhouseResult.Content.Trim() -ne "1") {
    throw "ClickHouse SELECT 1 failed: $($clickhouseResult.Content)"
}

Write-Host "[3/5] Checking PostgreSQL readiness..."
docker exec $PostgresContainer pg_isready -U $PostgresUser -d $PostgresDb | Out-Host

Write-Host "[4/5] Checking MinIO health..."
Invoke-WebRequest -UseBasicParsing -Uri $MinioHealthUrl | Out-Null

Write-Host "[5/6] Checking MCP skeleton services..."
Invoke-WebRequest -UseBasicParsing -Uri "$AssetRegistryMcpUrl/health" | Out-Null
Invoke-WebRequest -UseBasicParsing -Uri "$OpsMcpUrl/health" | Out-Null
Invoke-WebRequest -UseBasicParsing -Uri "$AuditMcpUrl/health" | Out-Null

Write-Host "[6/6] Checking initialized schemas..."
docker exec $PostgresContainer psql -U $PostgresUser -d $PostgresDb -c "SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('registry','audit','ops') ORDER BY schema_name;" | Out-Host
Invoke-WebRequest -UseBasicParsing -Uri "$ClickHouseUrl/?query=SELECT%20name%20FROM%20system.databases%20WHERE%20name%20IN%20('l1_raw','l2_base','l3_feature','l4_model','l5_strategy','l6_backtest','l7_trading')%20ORDER%20BY%20name" | Select-Object -ExpandProperty Content | Write-Host

Write-Host "OK cloud-center Win10 health check passed"
