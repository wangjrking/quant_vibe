from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
EXECUTOR = (
    ROOT
    / "quant"
    / "main"
    / "strategy_library"
    / "production"
    / "prod_v260_10d_regime_warmup_all4key_v20260724"
    / "code_snapshot"
    / "juejin_frozen_executor.py"
)


def action_file(directory: Path, case_id: str, action_label: str | None) -> Path:
    matches = []
    for path in directory.glob(f"{case_id}_*.csv"):
        header = path.open("r", encoding="utf-8-sig").readline()
        if (
            "action" in header
            and "stock_code" in header
            and (not action_label or action_label in path.name)
        ):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"expected one action file for {case_id}, got {matches}")
    return matches[0]


def parse_indicator(output: str) -> dict:
    prefix = "GM_BACKTEST_INDICATOR:"
    lines = [line for line in output.splitlines() if prefix in line]
    if not lines:
        raise RuntimeError("掘金日志未返回 GM_BACKTEST_INDICATOR")
    raw = lines[-1].split(prefix, 1)[1].strip()
    cleaned = raw.replace(
        "datetime.datetime(", "dict(__datetime_args__=("
    )
    try:
        return ast.literal_eval(raw)
    except Exception:
        keys = (
            "pnl_ratio",
            "pnl_ratio_annual",
            "sharp_ratio",
            "max_drawdown",
            "open_count",
            "close_count",
            "win_ratio",
            "calmar_ratio",
        )
        result = {}
        for key in keys:
            marker = f"'{key}': "
            if marker not in raw:
                continue
            value = raw.split(marker, 1)[1].split(",", 1)[0].strip()
            result[key] = ast.literal_eval(value)
        if not result:
            raise RuntimeError(f"无法解析掘金指标：{cleaned[:200]}")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="运行五只持仓冻结候选掘金回测")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--log-name", required=True)
    parser.add_argument("--action-label")
    args = parser.parse_args()

    report_dir = Path(args.report_dir).resolve()
    actions = action_file(report_dir, args.case_id, args.action_label)
    runtime = report_dir / f"juejin_runtime_{args.case_id}"
    runtime.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXECUTOR, runtime / "main.py")
    env = os.environ.copy()
    env.update(
        {
            "GM_SIGNAL_FILE": str(actions),
            "GM_BACKTEST_START": args.start,
            "GM_BACKTEST_END": args.end,
            "GM_BACKTEST_INITIAL_CASH": "700000",
            "GM_BACKTEST_SLIPPAGE_RATIO": "0.003",
        }
    )
    completed = subprocess.run(
        [str(PYTHON), "main.py"],
        cwd=runtime,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    log_path = report_dir / args.log_name
    log_path.write_text(
        completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"掘金回测失败 returncode={completed.returncode}: {log_path}")
    indicator = parse_indicator(completed.stdout)
    result = {
        "case_id": args.case_id,
        "action_file": str(actions),
        "start": args.start,
        "end": args.end,
        "slippage_each_side": 0.003,
        "commission_each_side": 0.0003,
        "sell_stamp_tax": 0.0005,
        "indicator": indicator,
        "log_path": str(log_path),
        "returncode": completed.returncode,
    }
    log_stem = Path(args.log_name).stem
    result_path = report_dir / f"{args.case_id}_{log_stem}_掘金结果.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
