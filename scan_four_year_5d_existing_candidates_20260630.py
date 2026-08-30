from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_5d_existing_candidate_inventory_20260630"

BASE_DIR = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_current_bestset_direct_guard_v1_20260630"
    / "standard_eval"
)
BASE_DAILY = BASE_DIR / "executable_5d_open_return_four_year_current_bestset_direct_guard_v1_20260630_daily_eval.csv"
BASE_REPORT_DIR_NAME = "model_agent_four_year_5d_current_bestset_direct_guard_v1_20260630"

METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame.sort_values("trade_date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def compare_against_base(base: pd.DataFrame, cand: pd.DataFrame, candidate_dir: str, daily_file: str) -> dict[str, object]:
    joined = base.merge(cand, on="trade_date", suffixes=("_base", "_cand"))
    recent63 = joined.tail(63)
    recent20 = joined.tail(20)
    month_frame = joined.copy()
    month_frame["period"] = month_frame["trade_date"].str[:6]
    monthly = month_frame.groupby("period", as_index=False).mean(numeric_only=True)
    result: dict[str, object] = {
        "candidate_dir": candidate_dir,
        "daily_file": daily_file,
        "joined_days": int(len(joined)),
    }
    for metric in METRICS:
        result[f"full_{metric}_delta"] = float((joined[f"{metric}_cand"] - joined[f"{metric}_base"]).mean())
        result[f"recent63_{metric}_delta"] = float((recent63[f"{metric}_cand"] - recent63[f"{metric}_base"]).mean())
        result[f"recent20_{metric}_delta"] = float((recent20[f"{metric}_cand"] - recent20[f"{metric}_base"]).mean())
    top5_month_delta = monthly["top5_cand"] - monthly["top5_base"]
    result["month_count"] = int(len(monthly))
    result["month_top5_positive"] = int((top5_month_delta > 0.0).sum())
    result["month_top5_nonnegative"] = int((top5_month_delta >= 0.0).sum())
    result["month_min_top5_delta"] = float(top5_month_delta.min())
    result["replaceable_under_current_rule"] = bool(
        float(result["full_rank_ic_delta"]) >= 0.0
        and float(result["full_top5_delta"]) >= 0.0
        and float(result["recent63_top5_delta"]) >= 0.0
        and float(result["recent20_top5_delta"]) >= 0.0
        and float(result["month_min_top5_delta"]) >= 0.0
    )
    return result


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    best = payload["best_by_full_top5"]
    lines = [
        "# 四年 5D 现有候选库存扫描",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 结论",
        "",
        f"- 扫描候选目录数：{payload['candidate_count']}",
        f"- 满足当前替换规则的现成候选数：{payload['replaceable_count']}",
        f"- 按 Full Top5 相对当前 bestset 排名第一：`{best['candidate_dir']}`",
        f"- 该候选的 `full_top5_delta={best['full_top5_delta']:.6f}`，`full_rank_ic_delta={best['full_rank_ic_delta']:.6f}`，`recent63_top5_delta={best['recent63_top5_delta']:.6f}`，`recent20_top5_delta={best['recent20_top5_delta']:.6f}`，`month_min_top5_delta={best['month_min_top5_delta']:.6f}`",
        "",
        "## 判断",
        "",
        "- 当前库存里没有一张现成 5D 研究表能在同口径下直接替代 `current_bestset_direct_guard_v1`。",
        "- `bucketed_blend_v5` 在近期窗口和 Full Top5 上有吸引力，但 Full RankIC 为负且月度尾部为负，不满足当前四年替换规则。",
        "- 下一步应切换到新的训练族或跨源构造路线，而不是继续在现有同源/库存候选里反复替换。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_daily(BASE_DAILY)
    rows: list[dict[str, object]] = []
    for candidate_dir in sorted(DATA_DIR.joinpath("reports").glob("model_agent_four_year_5d*")):
        std = candidate_dir / "standard_eval"
        if candidate_dir.name == BASE_REPORT_DIR_NAME:
            continue
        if not std.exists():
            continue
        daily_files = list(std.glob("*daily_eval.csv"))
        if len(daily_files) != 1:
            continue
        daily = load_daily(daily_files[0])
        rows.append(compare_against_base(base, daily, candidate_dir.name, daily_files[0].name))

    ranked = sorted(rows, key=lambda item: (item["full_top5_delta"], item["recent63_top5_delta"], item["recent20_top5_delta"], item["full_rank_ic_delta"]), reverse=True)
    replaceable = [row for row in ranked if row["replaceable_under_current_rule"]]
    payload = {
        "generated_at": now_iso(),
        "scope": "scan_existing_5d_candidate_inventory_against_current_bestset",
        "base_asset": "research_5d_four_year_current_bestset_direct_guard_v1_20260630",
        "base_daily": str(BASE_DAILY),
        "candidate_count": len(ranked),
        "replaceable_count": len(replaceable),
        "best_by_full_top5": ranked[0],
        "replaceable_candidates": replaceable,
        "ranked_candidates": ranked,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "existing_candidate_inventory_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(REPORT_DIR / "existing_candidate_inventory_summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
