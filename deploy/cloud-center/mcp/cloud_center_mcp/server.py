from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .service_catalog import SERVICE_CATALOG, get_service_definition


SERVICE_NAME = os.environ.get("MCP_SERVICE_NAME", "asset-registry-mcp")
SERVICE_PORT = int(os.environ.get("MCP_SERVICE_PORT", "8101"))
SERVICE_DEFINITION = get_service_definition(SERVICE_NAME)


def response_payload(path: str) -> tuple[int, dict[str, Any]]:
    if path == "/health":
        return 200, {
            "status": "ok",
            "service": SERVICE_NAME,
            "stage": SERVICE_DEFINITION.get("status", "skeleton"),
        }
    if path == "/metadata":
        return 200, {
            "service": SERVICE_NAME,
            **SERVICE_DEFINITION,
            "boundary": "health_and_metadata_skeleton_only",
        }
    if path == "/services":
        return 200, SERVICE_CATALOG
    if path == "/tools":
        return 200, {
            "service": SERVICE_NAME,
            "planned_tools": SERVICE_DEFINITION.get("planned_tools", []),
            "implemented_tools": [],
            "boundary": "真实读写工具尚未实现，不得用于生产资产访问。",
        }
    return 404, {
        "error": "not_found",
        "available_paths": ["/health", "/metadata", "/services", "/tools"],
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        status, payload = response_payload(parsed.path)
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> int:
    server = ThreadingHTTPServer(("0.0.0.0", SERVICE_PORT), Handler)
    print(f"Starting {SERVICE_NAME} skeleton on 0.0.0.0:{SERVICE_PORT}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
