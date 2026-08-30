from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd

import research_10d_exante_score_state_gate_v65_20260627 as gate
import research_four_year_10d_monthsafe_blend_gate_v4_20260629 as v4


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_10d_sparse_triple_gate_v6_20260629"

TOP_SINGLE_LIMIT = 35


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def choose_decision(pass_hard_count: int) -> str:
    if pass_hard_count > 0:
        return "sparse_triple_gate_v6_has_hard_pass_candidate_needs_review"
    return "continue_research_no_sparse_triple_hard_pass"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    control_daily = v4.load_daily(v4.CONTROL_DAILY)
    base_daily = v4.load_daily(v4.BASE_DAILY)
    cand_daily = v4.load_daily(v4.CAND_DAILY)
    common_dates = sorted(set(control_daily["trade_date"]) & set(base_daily["trade_date"]) & set(cand_daily["trade_date"]))
    control_daily = control_daily[control_daily["trade_date"].isin(common_dates)].sort_values("trade_date").reset_index(drop=True)
    base_daily = base_daily[base_daily["trade_date"].isin(common_dates)].sort_values("trade_date").reset_index(drop=True)
    cand_daily = cand_daily[cand_daily["trade_date"].isin(common_dates)].sort_values("trade_date").reset_index(drop=True)

    control_summary = gate.summarize(control_daily)
    base_summary = gate.summarize(base_daily)

    if v4.FEATURE_PATH.exists():
        features = pd.read_csv(v4.FEATURE_PATH)
        features["trade_date"] = features["trade_date"].astype(str)
    else:
        features = gate.score_state_features(gate.read_scores(v4.BASE_TABLE), gate.read_scores(v4.CAND_TABLE))
        features.to_csv(v4.FEATURE_PATH, index=False, encoding="utf-8-sig")

    features = features[features["trade_date"].isin(common_dates)].sort_values("trade_date").reset_index(drop=True)
    valid_dates = set(features["trade_date"])
    control_daily = control_daily[control_daily["trade_date"].isin(valid_dates)].sort_values("trade_date").reset_index(drop=True)
    base_daily = base_daily[base_daily["trade_date"].isin(valid_dates)].sort_values("trade_date").reset_index(drop=True)
    cand_daily = cand_daily[cand_daily["trade_date"].isin(valid_dates)].sort_values("trade_date").reset_index(drop=True)
    cand_diff = cand_daily[v4.METRICS] - base_daily[v4.METRICS]

    single_results = []
    for name, mask in gate.single_candidate_masks(features):
        if not bool(mask.any()):
            continue
        row, _daily, _month_control, _month_base = v4.evaluate_mask(
            name,
            mask,
            control_daily,
            base_daily,
            cand_diff,
            control_summary,
            base_summary,
        )
        single_results.append((row, name, mask))

    top_single = sorted(
        single_results,
        key=lambda x: (bool(x[0]["pass_hard"]), float(x[0]["objective"])),
        reverse=True,
    )[:TOP_SINGLE_LIMIT]

    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_daily: pd.DataFrame | None = None
    best_month_control: pd.DataFrame | None = None
    best_month_base: pd.DataFrame | None = None

    for (_, name_a, mask_a), (_, name_b, mask_b), (_, name_c, mask_c) in combinations(top_single, 3):
        mask = mask_a & mask_b & mask_c
        active_days = int(mask.sum())
        if active_days <= 0:
            continue
        row, daily, month_control_df, month_base_df = v4.evaluate_mask(
            f"({name_a}) AND ({name_b}) AND ({name_c})",
            mask,
            control_daily,
            base_daily,
            cand_diff,
            control_summary,
            base_summary,
        )
        rows.append(row)
        if best_row is None or (bool(row["pass_hard"]), float(row["objective"])) > (
            bool(best_row["pass_hard"]),
            float(best_row["objective"]),
        ):
            best_row = row
            best_daily = daily
            best_month_control = month_control_df
            best_month_base = month_base_df

    scan = pd.DataFrame(rows).sort_values(["pass_hard", "objective"], ascending=[False, False])
    scan_path = REPORT_DIR / "sparse_triple_gate_v6_scan.csv"
    scan.to_csv(scan_path, index=False, encoding="utf-8-sig")
    if best_daily is not None:
        best_daily.to_csv(REPORT_DIR / "best_daily_eval.csv", index=False, encoding="utf-8-sig")
    if best_month_control is not None:
        best_month_control.to_csv(REPORT_DIR / "best_monthly_delta_vs_control.csv", index=False, encoding="utf-8-sig")
    if best_month_base is not None:
        best_month_base.to_csv(REPORT_DIR / "best_monthly_delta_vs_base.csv", index=False, encoding="utf-8-sig")

    pass_hard_count = int(scan["pass_hard"].sum()) if not scan.empty else 0
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_four_year_10d_sparse_triple_gate_v6",
        "label": v4.LABEL,
        "base_table": v4.BASE_TABLE,
        "candidate_table": v4.CAND_TABLE,
        "top_single_limit": TOP_SINGLE_LIMIT,
        "scan_rows": int(len(scan)),
        "pass_hard_count": pass_hard_count,
        "best": best_row,
        "decision": choose_decision(pass_hard_count),
        "scan_csv": str(scan_path),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_prediction_table_write": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "sparse_triple_gate_v6_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
