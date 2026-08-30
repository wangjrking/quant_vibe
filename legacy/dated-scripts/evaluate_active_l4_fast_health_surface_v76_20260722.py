# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_fast_health_surface_v76_20260722"


def parse_indicator(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    line = next(line for line in reversed(text.splitlines()) if "GM_BACKTEST_INDICATOR:" in line)
    payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1]
    result = {}
    for key in ["pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio"]:
        match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
        result[key] = float(match.group(1))
    for key in ["open_count", "close_count"]:
        match = re.search(rf"'{key}':\s*([0-9]+)", payload)
        result[key] = int(match.group(1))
    return result


def main():
    protocol = json.loads((OUT / "preregistered_protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "juejin_action_manifest.json").read_text(encoding="utf-8"))
    targets = protocol["admission_targets"]
    rows = []
    for action in manifest["actions"]:
        case_id = action["representative_case_id"]
        values = parse_indicator(OUT / "juejin_logs" / f"{case_id}_slip0p003.log")
        passed = (
            values["pnl_ratio_annual"] >= targets["juejin_pnl_ratio_annual_linear_non_cagr_min"]
            and values["sharp_ratio"] >= targets["sharpe_min"]
            and values["max_drawdown"] <= targets["max_drawdown_max"]
        )
        rows.append({"case_id": case_id, "grid_points": action["grid_points"], "passed": passed, **values})

    passed_points = []
    for row in rows:
        if not row["passed"]:
            continue
        for point in row["grid_points"]:
            passed_points.append({"case_id": row["case_id"], **point})
    adjacent_pairs = []
    for index, left in enumerate(passed_points):
        for right in passed_points[index + 1 :]:
            distance = sum(
                abs(int(left[key]) - int(right[key]))
                for key in ["mean_index", "positive_index", "floor_index"]
            )
            if distance == 1:
                adjacent_pairs.append({"left": left, "right": right})

    passing_rows = [row for row in rows if row["passed"]]
    selected = sorted(
        passing_rows,
        key=lambda row: (
            -row["sharp_ratio"],
            -row["pnl_ratio_annual"],
            row["max_drawdown"],
            row["case_id"],
        ),
    )[0] if passing_rows else None
    observation_pass = (
        len(passed_points) >= int(targets["adjacent_pass_count_min"])
        and len(adjacent_pairs) > 0
    )
    payload = {
        "protocol_id": protocol["protocol_id"],
        "track": "research_only_observation",
        "platform": "juejin",
        "backtest_start": "2022-06-07 09:00:00",
        "backtest_end": "2025-12-31 15:30:00",
        "execution_price": "raw_unadjusted_open",
        "backtest_adjust": 0,
        "initial_cash": 700000,
        "slippage_ratio_each_side": 0.003,
        "annual_metric_semantics": "pnl_ratio_annual_linear_non_cagr",
        "known_2026_used": False,
        "results": rows,
        "pass_count": len(passed_points),
        "adjacent_pairs": adjacent_pairs,
        "observation_gate_pass": observation_pass,
        "selected_case": selected,
        "allow_historical_validation_open": observation_pass,
        "allow_production_promotion": False,
        "production_changed": False,
    }
    (OUT / "juejin_result_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected_text = selected["case_id"] if selected else "无"
    report = [
        "# V76 观察期掘金结论",
        "",
        f"- 严格通过点：`{len(passed_points)}` 个。",
        f"- 相邻通过对：`{len(adjacent_pairs)}` 对。",
        f"- 观察期门：`{'通过' if observation_pass else '未通过'}`。",
        f"- 冻结唯一候选：`{selected_text}`。",
        "- 本轮未读取2026进行参数选择，未修改生产。",
    ]
    (OUT / "观察期掘金结论.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"pass_count": len(passed_points), "adjacent_pairs": len(adjacent_pairs), "observation_gate_pass": observation_pass, "selected_case": selected_text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
