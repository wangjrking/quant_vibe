from __future__ import annotations


SERVICE_CATALOG: dict[str, dict[str, object]] = {
    "asset-registry-mcp": {
        "port": 8101,
        "owner": "data-warehouse-agent",
        "status": "skeleton",
        "description": "资产、manifest、lineage 和审计状态查询服务。",
        "planned_tools": [
            "get_asset",
            "list_assets",
            "get_manifest",
            "list_lineage",
            "check_asset_audit_status",
        ],
    },
    "ops-mcp": {
        "port": 8102,
        "owner": "commander-agent",
        "status": "skeleton",
        "description": "任务状态、部署状态、健康状态和运维日报查询服务。",
        "planned_tools": [
            "get_workflow_task",
            "list_open_tasks",
            "get_service_health",
            "record_deployment_event",
        ],
    },
    "audit-mcp": {
        "port": 8103,
        "owner": "audit-agent",
        "status": "skeleton",
        "description": "审计记录和证据路径查询服务。",
        "planned_tools": [
            "get_audit_record",
            "list_audit_records",
            "check_production_approval",
        ],
    },
    "l2-base-mcp": {
        "port": 8112,
        "owner": "data-integration-agent",
        "status": "skeleton",
        "description": "L2 综合底表覆盖、分区和只读查询服务。",
        "planned_tools": [
            "get_stock_daily_coverage",
            "query_stock_daily_window",
            "list_l2_assets",
        ],
    },
    "l3-feature-mcp": {
        "port": 8113,
        "owner": "factor-agent",
        "status": "skeleton",
        "description": "L3 因子、标签和 manifest 只读查询服务。",
        "planned_tools": [
            "get_factor_coverage",
            "query_factor_values",
            "list_label_assets",
        ],
    },
    "l4-model-mcp": {
        "port": 8114,
        "owner": "model-agent",
        "status": "skeleton",
        "description": "L4 预测资产、模型评分和 formal manifest 查询服务。",
        "planned_tools": [
            "get_prediction_manifest",
            "query_predictions",
            "check_approved_for_l5",
        ],
    },
}


def get_service_definition(service_name: str) -> dict[str, object]:
    default = {
        "port": 8100,
        "owner": "data-warehouse-agent",
        "status": "unknown",
        "description": "未登记的 cloud-center MCP 服务。",
        "planned_tools": [],
    }
    return SERVICE_CATALOG.get(service_name, default)
