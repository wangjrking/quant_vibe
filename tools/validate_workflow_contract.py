from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from workflow_contract import validate_layer_handoff_contract


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a standard L1-L8 workflow layer handoff contract JSON.")
    parser.add_argument("--file", required=True, help="Path to the contract JSON file.")
    parser.add_argument("--expected-layer", default=None)
    parser.add_argument("--expected-owner-agent", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = Path(args.file).resolve()
    payload = load_json(path)
    errors = validate_layer_handoff_contract(
        payload,
        expected_layer=args.expected_layer,
        expected_owner_agent=args.expected_owner_agent,
    )
    result = {
        "file": str(path),
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
