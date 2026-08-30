from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parent
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from workflow_contract_adapters import build_l4_contract_from_report
from workflow_contract import validate_layer_handoff_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 L4 报告生成标准工作流交接合同。")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--validation-output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    evidence = [str(args.report)] + [str(path) for path in args.evidence]
    contract = build_l4_contract_from_report(
        report,
        workflow_run_id=args.workflow_run_id,
        evidence_paths=evidence,
    )
    errors = validate_layer_handoff_contract(
        contract,
        expected_layer="L4",
        expected_owner_agent="model-agent",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = {
        "contract_path": str(args.output),
        "validator": "workflow_contract.validate_layer_handoff_contract",
        "expected_layer": "L4",
        "expected_owner_agent": "model-agent",
        "valid": not errors,
        "errors": errors,
        "ready_for_audit_review": contract["ready_for_audit_review"],
        "allow_next_layer_continue": contract["allow_next_layer_continue"],
    }
    args.validation_output.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
