from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from workflow_contract_adapters import (
    build_l1_contract_from_report,
    build_l2_contract_from_report,
    build_l3_contract_from_report,
    build_l4_contract_from_report,
    build_l5_contract_from_report,
    build_l6_contract_from_report,
    build_l7_contract_from_report,
    build_l8_contract_from_report,
    validate_built_contract,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a standard workflow layer handoff contract from an existing layer report.")
    parser.add_argument("--layer", required=True, choices=["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"])
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report_path = Path(args.report_file).resolve()
    output_path = Path(args.output).resolve()
    report = load_json(report_path)
    evidence_paths = [str(report_path), *args.evidence]

    builders = {
        "L1": build_l1_contract_from_report,
        "L2": build_l2_contract_from_report,
        "L3": build_l3_contract_from_report,
        "L4": build_l4_contract_from_report,
        "L5": build_l5_contract_from_report,
        "L6": build_l6_contract_from_report,
        "L7": build_l7_contract_from_report,
        "L8": build_l8_contract_from_report,
    }
    payload = builders[args.layer](
        report,
        workflow_run_id=args.workflow_run_id,
        evidence_paths=evidence_paths,
    )

    errors = validate_built_contract(payload, expected_layer=args.layer)
    result = {
        "valid": not errors,
        "errors": errors,
        "output": str(output_path),
    }
    if errors:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
