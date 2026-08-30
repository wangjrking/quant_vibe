from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_v260_style_risk_overlay_20260730"
PROTOCOL = OUT / "preregistered_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def board_of(stock_code: str) -> str:
    code = str(stock_code)
    if code.startswith("688"):
        return "star"
    if code.startswith(("300", "301")):
        return "chinext"
    return "main"


def transform_actions(
    actions: pd.DataFrame,
    single_name_cap: float,
    star_cap: float,
    growth_cap: float,
    min_entry_target: float,
) -> tuple[pd.DataFrame, dict]:
    active_target: dict[str, float] = {}
    rows: list[dict] = []
    skipped: list[dict] = []
    scaled: list[dict] = []

    for row in actions.to_dict("records"):
        stock_code = str(row["stock_code"])
        action = str(row["action"]).upper()
        if action == "SELL":
            active_target.pop(stock_code, None)
            rows.append(row)
            continue

        board = board_of(stock_code)
        original_target = float(row["target_pct"])
        target = min(original_target, single_name_cap)
        star_used = sum(
            value
            for code, value in active_target.items()
            if board_of(code) == "star"
        )
        growth_used = sum(
            value
            for code, value in active_target.items()
            if board_of(code) in {"star", "chinext"}
        )
        if board == "star":
            target = min(target, max(star_cap - star_used, 0.0))
        if board in {"star", "chinext"}:
            target = min(target, max(growth_cap - growth_used, 0.0))

        if target + 1e-12 < min_entry_target:
            skipped.append(
                {
                    "buy_date": str(row["buy_date"]),
                    "stock_code": stock_code,
                    "board": board,
                    "original_target_pct": original_target,
                    "available_target_pct": target,
                }
            )
            continue

        if target + 1e-12 < original_target:
            scaled.append(
                {
                    "buy_date": str(row["buy_date"]),
                    "stock_code": stock_code,
                    "board": board,
                    "original_target_pct": original_target,
                    "new_target_pct": target,
                }
            )
        row["target_pct"] = target
        active_target[stock_code] = target
        rows.append(row)

    transformed = pd.DataFrame(rows, columns=actions.columns)
    return transformed, {
        "input_rows": int(len(actions)),
        "output_rows": int(len(transformed)),
        "buy_rows": int((transformed["action"] == "BUY").sum()),
        "sell_rows": int((transformed["action"] == "SELL").sum()),
        "scaled_buy_rows": len(scaled),
        "skipped_buy_rows": len(skipped),
        "scaled_examples": scaled[:20],
        "skipped_examples": skipped[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(PROTOCOL))
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    output_root = protocol_path.parent
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_research_only":
        raise RuntimeError("Protocol is not frozen_research_only")

    action_path = ROOT / protocol["base_action_path"]
    actual_hash = sha256(action_path)
    if actual_hash != protocol["base_action_sha256"]:
        raise RuntimeError(
            f"Base action hash drift: expected={protocol['base_action_sha256']} "
            f"actual={actual_hash}"
        )

    actions = pd.read_csv(
        action_path,
        dtype={"signal_date": str, "buy_date": str, "stock_code": str},
    )
    output_dir = output_root / "actions"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for case in protocol["cases"]:
        transformed, audit = transform_actions(
            actions=actions,
            single_name_cap=float(case["single_name_cap"]),
            star_cap=float(case["star_cap"]),
            growth_cap=float(case["growth_cap"]),
            min_entry_target=float(protocol["min_entry_target"]),
        )
        output_path = output_dir / f"{case['case_id']}.csv"
        transformed.to_csv(output_path, index=False, encoding="utf-8-sig")
        manifest_rows.append(
            {
                **case,
                **audit,
                "action_path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
                "action_sha256": sha256(output_path),
            }
        )

    manifest = {
        "status": "research_only_actions_frozen",
        "protocol_path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(protocol_path),
        "base_action_path": protocol["base_action_path"],
        "base_action_sha256": actual_hash,
        "cases": manifest_rows,
        "production_changed": False,
        "formal_signal_generated": False,
    }
    (output_root / "action_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"cases": len(manifest_rows), "output": str(output_root)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
