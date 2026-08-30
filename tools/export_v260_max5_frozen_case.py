from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT / "quant" / "main"


def candidate_list(payload: dict) -> list[dict]:
    for value in payload.values():
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "case_id" in value[0]
            and "definition" in value[0]
        ):
            return value
    raise RuntimeError("冻结候选文件中未找到候选列表")


def find_frozen(directory: Path) -> Path:
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidate_list(payload)
            return path
        except Exception:
            continue
    raise RuntimeError("未找到观察期冻结候选 JSON")


def window(protocol: dict, start: str) -> dict:
    for value in protocol.values():
        if isinstance(value, dict) and value.get(next(iter(value), "")) == start:
            return value
    for value in protocol.values():
        if isinstance(value, dict) and start in value.values():
            return value
    raise RuntimeError(f"未找到窗口 {start}")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出五只持仓任意冻结候选动作")
    parser.add_argument("--round", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()

    os.environ["V260_MAX5_RESEARCH_ROUND"] = args.round
    sys.path.insert(0, str(MAIN))
    import research_v260_max5_observation_validation_20260727 as research

    report_dir = Path(args.report_dir).resolve()
    frozen = json.loads(find_frozen(report_dir).read_text(encoding="utf-8"))
    matches = [
        item for item in candidate_list(frozen) if item["case_id"] == args.case_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"冻结候选中未唯一找到 {args.case_id}")
    arrays, protocol, score, order = research.load_inputs()
    ranges = {
        "观察期": ("20220607", "20251231"),
        "验证期": ("20260105", "20260721"),
    }
    outputs = {}
    for label, (start, end) in ranges.items():
        daily, actions = research.run_case(
            arrays,
            score,
            order,
            matches[0]["definition"],
            protocol,
            start,
            end,
            record_actions=True,
        )
        daily_path = report_dir / f"{args.case_id}_{label}净值.csv"
        action_path = report_dir / f"{args.case_id}_{label}动作.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        actions.to_csv(action_path, index=False, encoding="utf-8-sig")
        outputs[label] = {
            "daily": str(daily_path),
            "actions": str(action_path),
            "action_rows": len(actions),
        }
    manifest = {
        "case_id": args.case_id,
        "definition": matches[0]["definition"],
        "outputs": outputs,
        "research_only": True,
    }
    output = report_dir / f"{args.case_id}_冻结导出清单.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
