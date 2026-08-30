from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd

import research_10d_exante_score_state_gate_v65_20260627 as gate


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"


SPECS = {
    "5d": {
        "label": "executable_5d_open_return",
        "base_daily": DATA_DIR / "reports" / "model_agent_four_year_control_assets_20260627" / "standard_eval" / "executable_5d_open_return_four_year_control_daily_eval.csv",
        "cand_daily": DATA_DIR / "reports" / "model_agent_four_year_gated_fusion_scan_20260627" / "standard_eval" / "executable_5d_open_return_four_year_gated_fusion_daily_eval.csv",
        "base_table": "stock_predict_data_model_agent_four_year_control_20260627_executable_5d_open_return_research",
        "cand_table": "stock_predict_data_model_agent_four_year_gated_fusion_20260627_executable_5d_open_return_research",
        "report_dir": DATA_DIR / "reports" / "model_agent_four_year_5d_recent_gate_scan_20260628",
        "min_full_rank_ic_delta": -0.001,
    },
    "10d": {
        "label": "executable_10d_open_return",
        "base_daily": DATA_DIR / "reports" / "model_agent_four_year_control_assets_20260627" / "standard_eval" / "executable_10d_open_return_four_year_control_daily_eval.csv",
        "cand_daily": DATA_DIR / "reports" / "model_agent_four_year_gated_fusion_scan_20260627" / "standard_eval" / "executable_10d_open_return_four_year_gated_fusion_daily_eval.csv",
        "base_table": "stock_predict_data_model_agent_four_year_control_20260627_executable_10d_open_return_research",
        "cand_table": "stock_predict_data_model_agent_four_year_gated_fusion_20260627_executable_10d_open_return_research",
        "report_dir": DATA_DIR / "reports" / "model_agent_four_year_10d_recent_gate_scan_20260628",
        "min_full_rank_ic_delta": -0.0005,
    },
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan four-year recent-window gates for 5D/10D candidates.")
    parser.add_argument("--target", choices=sorted(SPECS), required=True)
    return parser.parse_args()


def better(row: dict[str, object], best: dict[str, object] | None) -> bool:
    if best is None:
        return True
    return (bool(row["pass_hard"]), float(row["objective"])) > (bool(best["pass_hard"]), float(best["objective"]))


def pass_hard(row: dict[str, object], *, min_full_rank_ic_delta: float) -> bool:
    return (
        float(row["full_rank_ic_delta"]) >= min_full_rank_ic_delta
        and float(row["full_top5_delta"]) >= 0.0
        and float(row["recent63_top5_delta"]) >= 0.0
        and float(row["recent20_top5_delta"]) >= 0.0
        and int(row["positive_top5_months"]) >= 3
    )


def evaluate_mask(
    name: str,
    mask: pd.Series,
    base_daily: pd.DataFrame,
    cand_diff: pd.DataFrame,
    base_summary: dict[str, dict[str, float]],
    *,
    min_full_rank_ic_delta: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    daily = gate.make_daily_from_mask(base_daily, cand_diff, mask)
    candidate_summary = gate.summarize(daily)
    d = gate.delta(candidate_summary, base_summary)
    month_df, mstats = gate.monthly_stats(daily, base_daily)
    row = {
        "condition": name,
        "active_days": int(mask.sum()),
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
    }
    row["pass_hard"] = pass_hard(row, min_full_rank_ic_delta=min_full_rank_ic_delta)
    row["objective"] = (
        200.0 * float(row["recent20_top5_delta"])
        + 120.0 * float(row["recent63_top5_delta"])
        + 100.0 * float(row["recent20_top1_delta"])
        + 60.0 * float(row["recent63_top1_delta"])
        + 80.0 * float(row["recent20_rank_ic_delta"])
        + 50.0 * float(row["recent63_rank_ic_delta"])
        + 25.0 * float(row["full_top5_delta"])
        + 10.0 * float(row["full_rank_ic_delta"])
        - 20.0 * max(0.0, -float(row["min_month_top5_delta"]))
    )
    return row, daily, month_df


def main() -> int:
    args = parse_args()
    spec = SPECS[args.target]
    report_dir: Path = spec["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    base_daily = pd.read_csv(spec["base_daily"])
    cand_daily = pd.read_csv(spec["cand_daily"])
    for df in (base_daily, cand_daily):
        df["trade_date"] = df["trade_date"].astype(str)
        if "top50" not in df.columns:
            df["top50"] = df["top20"]
        df.sort_values("trade_date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    base_summary = gate.summarize(base_daily)
    raw_delta = gate.delta(gate.summarize(cand_daily), base_summary)

    feature_path = report_dir / "recent_gate_features.csv"
    if feature_path.exists():
        features = pd.read_csv(feature_path)
        features["trade_date"] = features["trade_date"].astype(str)
    else:
        features = gate.score_state_features(gate.read_scores(spec["base_table"]), gate.read_scores(spec["cand_table"]))
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
        row, daily, month_df = evaluate_mask(
            name,
            mask,
            base_daily,
            cand_diff,
            base_summary,
            min_full_rank_ic_delta=spec["min_full_rank_ic_delta"],
        )
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
        row, daily, month_df = evaluate_mask(
            f"({name_a}) AND ({name_b})",
            mask,
            base_daily,
            cand_diff,
            base_summary,
            min_full_rank_ic_delta=spec["min_full_rank_ic_delta"],
        )
        rows.append(row)
        if better(row, best_row):
            best_row = row
            best_daily = daily
            best_month = month_df

    scan = pd.DataFrame(rows).sort_values(["pass_hard", "objective"], ascending=[False, False])
    scan.to_csv(report_dir / "recent_gate_scan.csv", index=False, encoding="utf-8-sig")
    if best_daily is not None:
        best_daily.to_csv(report_dir / "best_daily_eval.csv", index=False, encoding="utf-8-sig")
    if best_month is not None:
        best_month.to_csv(report_dir / "best_monthly_delta.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": now_iso(),
        "scope": f"research_only_four_year_{args.target}_recent_gate_scan",
        "label": spec["label"],
        "base_table": spec["base_table"],
        "candidate_table": spec["cand_table"],
        "feature_source": "same-day prediction score distributions and rank differences only",
        "min_full_rank_ic_delta": spec["min_full_rank_ic_delta"],
        "raw_candidate_delta_vs_base": raw_delta,
        "scan_rows": int(len(scan)),
        "pass_hard_count": int(scan["pass_hard"].sum()) if not scan.empty else 0,
        "best": best_row,
        "decision": f"four_year_{args.target}_recent_gate_found" if best_row and best_row["pass_hard"] else f"continue_research_no_four_year_{args.target}_recent_gate_pass",
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
    (report_dir / "recent_gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
