from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_10d_month_segment_veto_v64_20260627"
BASE_DAILY = DATA_DIR / "reports" / "model_agent_current_best_promotion_review_20260627_v60" / "executable_10d_open_return_baseline_daily_eval.csv"
CAND_DAILY = DATA_DIR / "reports" / "model_agent_current_best_promotion_review_20260627_v60" / "executable_10d_open_return_candidate_daily_eval.csv"

METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def summarize(daily: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        "full": {m: float(daily[m].mean()) for m in METRICS},
        "recent126": {m: float(daily.tail(126)[m].mean()) for m in METRICS},
        "recent63": {m: float(daily.tail(63)[m].mean()) for m in METRICS},
        "recent20": {m: float(daily.tail(20)[m].mean()) for m in METRICS},
    }


def delta(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        scope: {metric: float(left[scope][metric] - right[scope][metric]) for metric in METRICS}
        for scope in ["full", "recent126", "recent63", "recent20"]
    }


def monthly_delta(candidate_daily: pd.DataFrame, base_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = candidate_daily.merge(base_daily, on="trade_date", suffixes=("", "_base"), validate="one_to_one")
    merged["month"] = pd.to_datetime(merged["trade_date"].astype(str), format="%Y%m%d").dt.to_period("M").astype(str)
    rows = []
    for month, group in merged.groupby("month", sort=True):
        rows.append(
            {
                "month": month,
                "rank_ic_delta": float(group["rank_ic"].mean() - group["rank_ic_base"].mean()),
                "top1_delta": float(group["top1"].mean() - group["top1_base"].mean()),
                "top5_delta": float(group["top5"].mean() - group["top5_base"].mean()),
            }
        )
    df = pd.DataFrame(rows)
    stats = {
        "month_count": int(len(df)),
        "positive_rank_ic_months": int((df["rank_ic_delta"] > 0).sum()),
        "positive_top1_months": int((df["top1_delta"] > 0).sum()),
        "positive_top5_months": int((df["top5_delta"] > 0).sum()),
        "nonnegative_top5_months": int((df["top5_delta"] >= 0).sum()),
        "positive_top1_and_nonnegative_top5_months": int(((df["top1_delta"] > 0) & (df["top5_delta"] >= 0)).sum()),
        "min_rank_ic_delta": float(df["rank_ic_delta"].min()),
        "min_top5_delta": float(df["top5_delta"].min()),
    }
    return df, stats


def make_daily(base: pd.DataFrame, cand: pd.DataFrame, veto_months: set[str]) -> pd.DataFrame:
    out = cand.copy()
    month = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d").dt.to_period("M").astype(str)
    use_base = month.isin(veto_months)
    for metric in METRICS:
        out.loc[use_base, metric] = base.loc[use_base, metric].to_numpy()
    return out


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_DAILY)
    cand = pd.read_csv(CAND_DAILY)
    base["trade_date"] = base["trade_date"].astype(str)
    cand["trade_date"] = cand["trade_date"].astype(str)
    base_summary = summarize(base)
    base_month, _ = monthly_delta(cand, base)
    negative_months = base_month.loc[base_month["top5_delta"] < 0, "month"].tolist()

    rows = []
    best_payload = None
    # Scan all subsets up to the full negative-month set. There are only five in v60.
    for size in range(1, len(negative_months) + 1):
        for subset in combinations(negative_months, size):
            veto = set(subset)
            daily = make_daily(base, cand, veto)
            cur_summary = summarize(daily)
            deltas = delta(cur_summary, base_summary)
            month_df, month_stats = monthly_delta(daily, base)
            objective = (
                3.0 * deltas["recent20"]["top1"]
                + 2.0 * deltas["recent63"]["top1"]
                + 1.2 * deltas["full"]["top1"]
                + 3.5 * deltas["full"]["top5"]
                + 2.0 * deltas["recent63"]["top5"]
                + 1.5 * deltas["recent20"]["top5"]
                + 0.08 * month_stats["positive_top5_months"]
                - 5.0 * max(0.0, -month_stats["min_top5_delta"])
            )
            row = {
                "veto_months": ",".join(sorted(veto)),
                "veto_count": len(veto),
                "objective": float(objective),
                "full_rank_ic_delta": deltas["full"]["rank_ic"],
                "full_top1_delta": deltas["full"]["top1"],
                "full_top5_delta": deltas["full"]["top5"],
                "recent126_top5_delta": deltas["recent126"]["top5"],
                "recent63_top1_delta": deltas["recent63"]["top1"],
                "recent63_top5_delta": deltas["recent63"]["top5"],
                "recent20_top1_delta": deltas["recent20"]["top1"],
                "recent20_top5_delta": deltas["recent20"]["top5"],
                "positive_top5_months": month_stats["positive_top5_months"],
                "nonnegative_top5_months": month_stats["nonnegative_top5_months"],
                "min_month_top5_delta": month_stats["min_top5_delta"],
                "pass_hard": bool(
                    deltas["full"]["top5"] >= 0
                    and deltas["recent63"]["top5"] >= 0
                    and deltas["recent20"]["top5"] >= 0
                    and month_stats["positive_top5_months"] >= 3
                    and month_stats["min_top5_delta"] >= 0
                    and deltas["full"]["rank_ic"] >= -0.0005
                ),
            }
            rows.append(row)
            if row["pass_hard"] and (best_payload is None or row["objective"] > best_payload["row"]["objective"]):
                best_payload = {
                    "row": row,
                    "daily": daily,
                    "monthly_delta": month_df,
                    "delta_vs_base": deltas,
                    "monthly_stats": month_stats,
                }

    result = pd.DataFrame(rows).sort_values(
        ["pass_hard", "objective", "positive_top5_months", "full_top1_delta"],
        ascending=[False, False, False, False],
    )
    result.to_csv(REPORT_DIR / "month_segment_veto_scan.csv", index=False, encoding="utf-8-sig")
    if best_payload:
        best_payload["daily"].to_csv(REPORT_DIR / "best_daily_eval.csv", index=False, encoding="utf-8-sig")
        best_payload["monthly_delta"].to_csv(REPORT_DIR / "best_monthly_delta.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_10d_month_segment_veto_v64",
        "label": "executable_10d_open_return",
        "negative_months_from_v60": negative_months,
        "scan_rows": int(len(result)),
        "pass_hard_count": int(result["pass_hard"].sum()) if len(result) else 0,
        "best": best_payload["row"] if best_payload else (result.iloc[0].to_dict() if len(result) else None),
        "best_delta_vs_base": best_payload["delta_vs_base"] if best_payload else None,
        "best_monthly_stats": best_payload["monthly_stats"] if best_payload else None,
        "decision": "diagnostic_month_segment_candidate_passed" if best_payload else "no_month_segment_candidate_passed",
        "note": "Month-segment veto uses label-informed month exclusion and is diagnostic-only unless converted into an ex-ante observable regime rule.",
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_prediction_write": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "month_segment_veto_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 10D 月份分段 veto 诊断 v64",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 结论",
        "",
        f"决策：`{summary['decision']}`。",
        "",
        "该扫描使用已知月度评价结果来选择禁用月份，因此只能作为诊断，不可直接作为生产规则或正式模型结论。",
        "",
        "## 边界",
        "",
        "- 未训练模型",
        "- 未写入新预测表",
        "- 未改 formal manifest",
        "- 未改 production manifest",
        "- 未生成交易信号",
        "- 未运行策略回测",
    ]
    (REPORT_DIR / "month_segment_veto_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "pass_hard_count": summary["pass_hard_count"], "best": summary["best"], "report_dir": str(REPORT_DIR)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
