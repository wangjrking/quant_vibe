from __future__ import annotations

import json
import shutil
from pathlib import Path

import archive_dynamic_h2_m3_sl06_dd0712_pos90_tp060_candidate_20260624 as base


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
STRATEGY_ID = "candidate_dynamic_h2m3_sl06dd0712_pos89_tp060_v20260624"
ARCHIVE_DIR = MAIN / "strategy_library" / "exploration" / STRATEGY_ID

BASE_REPORT_DIR = (
    DATA
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "top1_no_delist_rerun_20260624"
    / "diversification_tune_20260624"
    / "strict_sync_liquidity_neighborhood_20260624"
    / "formal_horizon_entry_confirmation_20260624"
)
VALIDATION_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_sl06_dd0712_pos89_tp060_validation_20260624"
POSITION_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_dd0712_tp060_position_20260624"
SIGNAL_FILE = POSITION_DIR / "signals" / "dd0712_tp060_pos89.csv"
STRATEGY_MAIN = BASE_REPORT_DIR.parent.parent / "strict_sync_strategy_snapshot_20260624" / "main.py"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def trading_rules() -> dict:
    rules = base.trading_rules()
    rules["strategy_id"] = STRATEGY_ID
    rules["strategy_name"] = "动态持有 Top1 三周期融合 pos89 收益回撤改善候选"
    rules["position_rule"]["target_position_pct"] = 0.89
    rules["execution_rule"]["signal_date_rule"] = "使用 T 日 formal L4 分数生成 T+1 开盘交易信号"
    return rules


def report_text(validation: dict, latest: dict) -> str:
    metrics = validation["metrics"]
    stress = validation["contribution_stress"]
    return "\n".join(
        [
            "# dd0712 pos89 收益回撤改善候选归档",
            "",
            "## 当前结论",
            "",
            "该版本是探索候选，不是生产策略。它只在 `dd0712_tp060` 规则上把目标仓位从 `90%` 调整为 `89%`，其余 formal L4 输入、Top1 选股、动态持有、止损止盈、账户回撤缩放和硬过滤口径保持一致。",
            "",
            "## 掘金结果",
            "",
            f"- 年化收益：`{float(metrics['annual_return']):.2%}`",
            f"- Sharpe：`{float(metrics['sharpe']):.2f}`",
            f"- 最大回撤：`{float(metrics['max_drawdown']):.2%}`",
            f"- recent60 年化：`{float(metrics['recent60_annual']):.2%}`",
            f"- recent120 年化：`{float(metrics['recent120_annual']):.2%}`",
            f"- 2026YTD 年化：`{float(metrics['ytd_annual']):.2%}`",
            "",
            "## 强准入状态",
            "",
            "- 硬过滤预检通过：北交所、ST/风险警示、退市、开盘涨停买入命中均为 `0`。",
            "- 低路径依赖预检通过：三个近期开仓锚点的最差年化均为正，且高于当前弱收益目标。",
            f"- 贡献压力预检通过：去最高 1% 代理信号后年化仍为 `{float(stress['drop_top_1pct']['annual_return']):.2%}`。",
            "- 尚未通过生产发布：仍需审计复核和用户批准，不得称为当前生产策略。",
            "",
            "## 最新信号覆盖",
            "",
            f"- 最新 signal_date：`{latest.get('signal_date')}`",
            f"- 最新 buy_date：`{latest.get('buy_date')}`",
            f"- 最新标的：`{latest.get('stock_code')}` / `{latest.get('name')}`",
            "",
            "## 证据路径",
            "",
            f"- 验证目录：`{VALIDATION_DIR}`",
            f"- 源信号：`{SIGNAL_FILE}`",
            f"- 归档目录：`{ARCHIVE_DIR}`",
            "",
        ]
    )


def main() -> int:
    if not VALIDATION_DIR.exists():
        raise FileNotFoundError(VALIDATION_DIR)
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(SIGNAL_FILE)

    base.STRATEGY_ID = STRATEGY_ID
    base.ARCHIVE_DIR = ARCHIVE_DIR
    base.VALIDATION_DIR = VALIDATION_DIR
    base.SIGNAL_FILE = SIGNAL_FILE

    summary = base.read_json(VALIDATION_DIR / "validation_summary.json")
    validation = base.validation_payload(summary)
    signal_rows = base.read_csv(SIGNAL_FILE)
    latest = base.latest_signal(signal_rows)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ARCHIVE_DIR / "prediction_table_meta.json", base.prediction_meta())
    write_json(ARCHIVE_DIR / "trading_rules.json", trading_rules())
    write_json(ARCHIVE_DIR / "validation.json", validation)
    write_json(
        ARCHIVE_DIR / "strategy_manifest.json",
        {
            "schema_version": 1,
            "strategy_id": STRATEGY_ID,
            "status": "exploration_candidate_pending_audit_not_published",
            "created_at": "2026-06-24",
            "validation_platform": "juejin",
            "source_signal_file": str(SIGNAL_FILE),
            "validation_dir": str(VALIDATION_DIR),
            "registry_updated": False,
            "production_published": False,
        },
    )
    (ARCHIVE_DIR / "report.md").write_text(report_text(validation, latest), encoding="utf-8")

    copy_file(SIGNAL_FILE, ARCHIVE_DIR / "signals" / SIGNAL_FILE.name)
    copy_file(VALIDATION_DIR / "validation_report.md", ARCHIVE_DIR / "backtests" / "validation_report.md")
    copy_file(VALIDATION_DIR / "validation_summary.json", ARCHIVE_DIR / "backtests" / "validation_summary.json")
    copy_file(VALIDATION_DIR / "time_slices.csv", ARCHIVE_DIR / "backtests" / "time_slices.csv")
    copy_file(VALIDATION_DIR / "nearby_summary.csv", ARCHIVE_DIR / "backtests" / "nearby_summary.csv")
    copy_file(VALIDATION_DIR / "stress_summary.csv", ARCHIVE_DIR / "backtests" / "stress_summary.csv")
    copy_file(VALIDATION_DIR / "hard_gate_audit.json", ARCHIVE_DIR / "backtests" / "hard_gate_audit.json")
    copy_file(POSITION_DIR / "summary.csv", ARCHIVE_DIR / "research_grid" / "position_summary.csv")
    copy_file(POSITION_DIR / "position_report.md", ARCHIVE_DIR / "research_grid" / "position_report.md")
    if STRATEGY_MAIN.exists():
        copy_file(STRATEGY_MAIN, ARCHIVE_DIR / "code_snapshot" / "main.py")
    copy_file(MAIN / "validate_dynamic_h2_m3_sl06_dd0712_pos89_tp060_20260624.py", ARCHIVE_DIR / "code_snapshot" / "validate_dynamic_h2_m3_sl06_dd0712_pos89_tp060_20260624.py")
    copy_file(Path(__file__), ARCHIVE_DIR / "code_snapshot" / Path(__file__).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
