$ErrorActionPreference = "Stop"

$ClickHouseUrl = if ($env:CLICKHOUSE_URL) { $env:CLICKHOUSE_URL } else { "http://127.0.0.1:8123" }
$MinioHealthUrl = if ($env:MINIO_HEALTH_URL) { $env:MINIO_HEALTH_URL } else { "http://127.0.0.1:9100/minio/health/live" }
$PostgresContainer = if ($env:POSTGRES_CONTAINER) { $env:POSTGRES_CONTAINER } else { "quant-postgres" }
$PostgresUser = if ($env:POSTGRES_USER) { $env:POSTGRES_USER } else { "quant_admin" }
$PostgresDb = if ($env:POSTGRES_DB) { $env:POSTGRES_DB } else { "quant_warehouse" }

Write-Host "[1/4] Checking ClickHouse SELECT 1..."
$clickhouseResult = Invoke-WebRequest -UseBasicParsing -Uri "$ClickHouseUrl/?query=SELECT%201"
if ($clickhouseResult.Content.Trim() -ne "1") {
    throw "ClickHouse SELECT 1 failed: $($clickhouseResult.Content)"
}

Write-Host "[2/4] Checking PostgreSQL readiness..."
docker exec $PostgresContainer pg_isready -U $PostgresUser -d $PostgresDb | Out-Host

Write-Host "[3/4] Checking MinIO health..."
Invoke-WebRequest -UseBasicParsing -Uri $MinioHealthUrl | Out-Null

Write-Host "[4/4] Checking Docker Compose services..."
docker compose --env-file /opt/quant_mcp/env/warehouse.env -f deploy/cloud-center/docker-compose.yml ps | Out-Host

Write-Host "OK cloud-center health check passed"
