#!/usr/bin/env bash
set -euo pipefail

CLICKHOUSE_URL="${CLICKHOUSE_URL:-http://127.0.0.1:8123}"
MINIO_HEALTH_URL="${MINIO_HEALTH_URL:-http://127.0.0.1:9100/minio/health/live}"
ASSET_REGISTRY_MCP_URL="${ASSET_REGISTRY_MCP_URL:-http://127.0.0.1:8101}"
OPS_MCP_URL="${OPS_MCP_URL:-http://127.0.0.1:8102}"
AUDIT_MCP_URL="${AUDIT_MCP_URL:-http://127.0.0.1:8103}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-quant-postgres}"
POSTGRES_USER="${POSTGRES_USER:-quant_admin}"
POSTGRES_DB="${POSTGRES_DB:-quant_warehouse}"

echo "[1/4] Checking ClickHouse SELECT 1..."
curl -fsS "${CLICKHOUSE_URL}/?query=SELECT%201" | grep -q 1

echo "[2/4] Checking PostgreSQL readiness..."
docker exec "${POSTGRES_CONTAINER}" pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null

echo "[3/4] Checking MinIO health..."
curl -fsS "${MINIO_HEALTH_URL}" >/dev/null

echo "[4/5] Checking MCP skeleton services..."
curl -fsS "${ASSET_REGISTRY_MCP_URL}/health" >/dev/null
curl -fsS "${OPS_MCP_URL}/health" >/dev/null
curl -fsS "${AUDIT_MCP_URL}/health" >/dev/null

echo "[5/5] Checking Docker Compose services..."
docker compose --env-file /opt/quant_mcp/env/warehouse.env -f deploy/cloud-center/docker-compose.yml ps

echo "OK cloud-center health check passed"
