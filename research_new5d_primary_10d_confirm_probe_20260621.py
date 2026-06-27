from __future__ import annotations

import json
from pathlib import Path

import research_5d_primary_10d_confirm_probe_20260621 as base


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_5D = ROOT / "quant" / "main" / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "new5d_primary_10d_confirm_probe_20260621"
)


def main() -> None:
    manifest = json.loads(MANIFEST_5D.read_text(encoding="utf-8"))
    if manifest.get("approval_status") != "approved_for_l5":
        raise SystemExit(f"5D manifest is not approved_for_l5: {MANIFEST_5D}")
    table = str(manifest["table"])
    base.TABLE_5D = table
    base.REPORT_DIR = REPORT_DIR
    base.main()


if __name__ == "__main__":
    main()
