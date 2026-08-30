from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd

import research_10d_exante_score_state_gate_v65_20260627 as gate


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d_exante_gate_20260627"
BASE_DAILY = DATA_DIR / "reports" / "model_agent_four_year_control_assets_20260627" / "standard_eval" / "executable_3d_open_return_four_year_control_daily_eval.csv"
CAND_DAILY = DATA_DIR / "reports" / "model_agent_four_year_adaptive_rank_fusion_vs_control_20260627" / "standard_eval" / "executable_3d_open_return_four_year_adaptive_rank_fusion_vs_control_daily_eval.csv"
BASE_TABLE = "stock_predict_data_model_agent_four_year_control_20260627_executable_3d_open_return_research"
CAND_TABLE = "stock_predict_data_model_agent_four_year_adaptive_rank_fusion_vs_control_20260627_executable_3d_open_return_research"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def better(row: dict[str, object], best: dict[str, object] | None) -> bool:
    if best is None:
        return True
    return (bool(row["pass_hard"]), float(row["objective"])) > (bool(best["pass_hard"]), float(best["objective"]))


def pass_hard_for_soft_risk_improvement(d: dict[str, dict[str, float]], mstats: dict[str, object]) -> bool:
    return (
        d["full"]["top5"] >= 0.0
        and d["full"]["rank_ic"] >= -0.0015
        and d["recent63"]["top5"] >= 0.0
        and int(mstats["positive_top5_months"]) >= 3
    )


def evaluate_mask(
    name: str,
    mask: pd.Series,
    base_daily: pd.DataFrame,
    cand_diff: pd.DataFrame,
    base_summary: dict[str, dict[str, float]],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    daily = gate.make_daily_from_mask(base_daily, cand_diff, mask)
    candidate_summary = gate.summarize(daily)
    d = gate.delta(candidate_summary, base_summary)
    month_df, mstats = gate.monthly_stats(daily, base_daily)
    hard = pass_hard_for_soft_risk_improvement(d, mstats)
    objective = (
        160.0 * d["recent63"]["rank_ic"]
        + 120.0 * d["recent20"]["rank_ic"]
        + 80.0 * d["full"]["rank_ic"]
        + 50.0 * d["recent63"]["top5"]
        + 30.0 * d["recent20"]["top5"]
        + 25.0 * d["full"]["top5"]
        + 10.0 * d["full"]["top1"]
        - 20.0 * max(0.0, -float(mstats["min_top5_delta"]))
    )
    row = {
        "condition": name,
        "active_days": int(mask.sum()),
        "objective": float(objective),
        "full_rank_ic_delta": d["full"]["rank_ic"],
        "full_top1_delta": d["full"]["top1"],
        "full_top3_delta": d["full"]["top3"],
        "full_top5_delta": d["full"]["top5"],
        "recent126_rank_ic_delta": d["recent126"]["rank_ic"],
        "recent126_top5_delta": d["recent126"]["top5"],
        "recent63_rank_ic_delta": d["recent63"]["rank_ic"],
        "recent63_top1_delta": d["recent63"]["top1"],
        "recent63_top5_delta": d["recent63"]["top5"],
        "recent20_rank_ic_delta": d["recent20"]["rank_ic"],
        "recent20_top1_delta": d["recent20"]["top1"],
        "recent20_top5_delta": d["recent20"]["top5"],
        "positive_top5_months": int(mstats["positive_top5_months"]),
        "nonnegative_top5_months": int(mstats["nonnegative_top5_months"]),
        "min_month_top5_delta": float(mstats["min_top5_delta"]),
        "pass_hard": bool(hard),
    }
    return row, daily, month_df


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    base_daily = pd.read_csv(BASE_DAILY)
    cand_daily = pd.read_csv(CAND_DAILY)
    for df in (base_daily, cand_daily):
        df["trade_date"] = df["trade_date"].astype(str)
        if "top50" not in df.columns:
            df["top50"] = df["top20"]
        df.sort_values("trade_date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    base_summary = gate.summarize(base_daily)
    raw_delta = gate.delta(gate.summarize(cand_daily), base_summary)

    feature_path = REPORT_DIR / "four_year_3d_exante_features.csv"
    if feature_path.exists():
        features = pd.read_csv(feature_path)
        features["trade_date"] = features["trade_date"].astype(str)
    else:
        features = gate.score_state_features(gate.read_scores(BASE_TABLE), gate.read_scores(CAND_TABLE))
        features.to_csv(feature_path, index=False)

    eval_dates = set(base_daily["trade_date"])
    features = features[features["trade_date"].isin(eval_dates)].sort_values("trade_date").reset_index(drop=True)
    valid_dates = set(features["trade_date"])
    base_daily = base_daily[base_daily["trade_date"].isin(valid_dates)].sort_values("trade_date").reset_index(drop=True)
    cand_daily = cand_daily[cand_daily["trade_date"].isin(valid_dates)].sort_values("trade_date").reset_index(drop=True)
    cand_diff = cand_daily[gate.METRICS] - base_daily[gate.METRICS]

    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_daily = None
    best_month = None
    single_results: list[tuple[dict[str, object], str, pd.Series]] = []

    for name, mask in gate.single_candidate_masks(features):
        if not bool(mask.any()):
            continue
        row, daily, month_df = evaluate_mask(name, mask, base_daily, cand_diff, base_summary)
        rows.append(row)
        single_results.append((row, name, mask))
        if better(row, best_row):
            best_row = row
            best_daily = daily
            best_month = month_df

    top_single = sorted(single_results, key=lambda x: (bool(x[0]["pass_hard"]), float(x[0]["objective"])), reverse=True)[:40]
    for (_, name_a, mask_a), (_, name_b, mask_b) in combinations(top_single, 2):
        mask = mask_a & mask_b
        if not bool(mask.any()):
            continue
        row, daily, month_df = evaluate_mask(f"({name_a}) AND ({name_b})", mask, base_daily, cand_diff, base_summary)
        rows.append(row)
        if better(row, best_row):
            best_row = row
            best_daily = daily
            best_month = month_df

    scan = pd.DataFrame(rows).sort_values(["pass_hard", "objective"], ascending=[False, False])
    scan.to_csv(REPORT_DIR / "four_year_3d_exante_gate_scan.csv", index=False, encoding="utf-8-sig")
    if best_daily is not None:
        best_daily.to_csv(REPORT_DIR / "best_daily_eval.csv", index=False, encoding="utf-8-sig")
    if best_month is not None:
        best_month.to_csv(REPORT_DIR / "best_monthly_delta.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_four_year_3d_exante_gate",
        "label": "executable_3d_open_return",
        "base_table": BASE_TABLE,
        "candidate_table": CAND_TABLE,
        "feature_source": "same-day prediction score distributions and rank differences only",
        "raw_candidate_delta_vs_base": raw_delta,
        "scan_rows": int(len(scan)),
        "pass_hard_count": int(scan["pass_hard"].sum()) if not scan.empty else 0,
        "best": best_row,
        "decision": "four_year_3d_exante_gate_found" if best_row and best_row["pass_hard"] else "continue_research_no_four_year_3d_exante_hard_pass",
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_prediction_write": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True
        }
    }
    (REPORT_DIR / "four_year_3d_exante_gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 四年观察期 3D 前视分数状态切换扫描",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 结论",
        "",
        f"- 决策：`{summary['decision']}`",
        f"- 硬门槛通过数量：`{summary['pass_hard_count']}` / `{summary['scan_rows']}`",
        f"- 最优条件：`{best_row['condition'] if best_row else '无'}`",
        "",
        "## 边界",
        "",
        "- 本次仅做 research-only 评分切换扫描",
        "- 未训练模型",
        "- 未写 formal manifest",
        "- 未切 production manifest",
        "- 未生成信号",
        "- 未跑回测",
    ]
    (REPORT_DIR / "four_year_3d_exante_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
