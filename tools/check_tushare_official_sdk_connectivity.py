from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_token(config_path: str | None) -> str:
    sys.path.insert(0, str(_project_root()))
    from project_paths import load_config

    config = load_config(config_path)
    token = config.get("datasource", {}).get("tushare_token")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    return str(token)


def _probe(endpoint: str | None, token: str, trade_date: str) -> dict:
    sys.path.insert(0, str(_project_root()))
    import tushare as ts
    from data_load_module import get_pro

    result = {
        "endpoint": endpoint or "default",
        "success": False,
        "shape": None,
        "error_type": None,
        "error_summary": None,
        "tushare_version": getattr(ts, "__version__", None),
    }
    try:
        pro = get_pro(token, endpoint=endpoint)
        frame = pro.query("daily", trade_date=trade_date)
        result["success"] = True
        result["shape"] = [int(frame.shape[0]), int(frame.shape[1])]
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic boundary.
        result["error_type"] = type(exc).__name__
        result["error_summary"] = str(exc)[:500]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check official Tushare SDK connectivity without writing business data."
    )
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        help="Optional official SDK endpoint to test. Can be repeated.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    endpoints = [None]
    endpoints.extend(args.endpoint)

    payload = {
        "tool": "check_tushare_official_sdk_connectivity",
        "trade_date": args.trade_date,
        "python": sys.executable,
        "checked_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "official_sdk_only": True,
        "business_data_written": False,
        "probes": [],
    }

    try:
        token = _load_token(args.config)
    except Exception as exc:  # noqa: BLE001
        payload["token_configured"] = False
        payload["success"] = False
        payload["error_type"] = type(exc).__name__
        payload["error_summary"] = str(exc)[:500]
    else:
        payload["token_configured"] = True
        for endpoint in endpoints:
            payload["probes"].append(_probe(endpoint, token, args.trade_date))
        payload["success"] = any(probe["success"] for probe in payload["probes"])

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
