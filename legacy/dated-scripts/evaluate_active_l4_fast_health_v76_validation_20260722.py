# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_fast_health_v76_validation_20260722"


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
    protocol = json.loads((OUT / "validation_protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "validation_action_manifest.json").read_text(encoding="utf-8"))
    gate = protocol["validation_gate"]
    observed = protocol["observation_summary"]
    annual_floor = float(observed["pnl_ratio_annual"]) * float(gate["annual_retention_ratio_min"])
    sharpe_floor = float(observed["sharp_ratio"]) * float(gate["sharpe_retention_ratio_min"])
    rows = []
    for action in manifest["actions"]:
        case_id = action["case_id"]
        values = parse_indicator(OUT / "juejin_logs" / f"{case_id}_slip0p003.log")
        passed = (
            values["pnl_ratio"] > 0.0
            and values["pnl_ratio_annual"] >= annual_floor
            and values["sharp_ratio"] >= sharpe_floor
            and values["max_drawdown"] <= float(gate["max_drawdown_max"])
        )
        rows.append(
            {
                "case_id": case_id,
                "profile_id": action["profile"]["id"],
                "start_signal_date": action["actual_start_signal_date"],
                "start_buy_date": action["backtest_start_buy_date"],
                "passed": passed,
                **values,
            }
        )
    main_rows = [row for row in rows if row["profile_id"] == "p_main"]
    primary_start = min(row["start_signal_date"] for row in rows)
    primary_rows = [row for row in rows if row["start_signal_date"] == primary_start]
    main_pass_count = sum(row["passed"] for row in main_rows)
    primary_profile_pass_count = sum(row["passed"] for row in primary_rows)
    validation_pass = (
        main_pass_count >= int(gate["main_profile_start_pass_min"])
        and primary_profile_pass_count >= int(gate["primary_start_profile_pass_min"])
    )
    payload = {
        "protocol_id": protocol["protocol_id"],
        "track": "research_only_historical_validation",
        "platform": "juejin",
        "annual_floor": annual_floor,
        "sharpe_floor": sharpe_floor,
        "max_drawdown_ceiling": gate["max_drawdown_max"],
        "results": rows,
        "main_profile_start_pass_count": main_pass_count,
        "primary_start_profile_pass_count": primary_profile_pass_count,
        "validation_gate_pass": validation_pass,
        "allow_production_promotion": False,
        "parameter_reselection_allowed": False,
        "production_changed": False,
    }
    (OUT / "validation_result_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    main_pnl = [row["pnl_ratio"] for row in main_rows]
    main_sharpe = [row["sharp_ratio"] for row in main_rows]
    report = [
        "# V76 历史验证结论",
        "",
        "## 结论",
        "",
        "验证未通过，候选必须淘汰，不得发布，也不得根据验证结果重新选择参数。",
        "",
        f"- 主参数五个空仓启动点通过数：`{main_pass_count}/5`。",
        f"- 首个启动点四个相邻参数通过数：`{primary_profile_pass_count}/4`。",
        f"- 主参数累计收益范围：`{min(main_pnl):.2%}` 至 `{max(main_pnl):.2%}`。",
        f"- 主参数 Sharpe 范围：`{min(main_sharpe):.4f}` 至 `{max(main_sharpe):.4f}`。",
        "- 20条验证路径全部未通过。",
        "- 未修改生产策略、registry 或正式信号。",
    ]
    (OUT / "历史验证结论.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"validation_gate_pass": validation_pass, "main_profile_start_pass_count": main_pass_count, "primary_start_profile_pass_count": primary_profile_pass_count, "paths": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
