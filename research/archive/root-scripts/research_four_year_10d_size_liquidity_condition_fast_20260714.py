from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "model_agent_four_year_10d_size_liquidity_condition_fast_20260714"
)
FEATURE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
TRAIN_END = "20251231"
HOLDOUT_START = "20260101"
LABEL_KEY = "10d"
LABEL_COL = "label_10d"
FEATURES = ["total_mv", "amount"]
THRESHOLDS = [0.2, 0.3, 0.7, 0.8]
BONUSES = [0.02, 0.05]

sys.path.insert(0, str(MAIN))
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402


def _load() -> pd.DataFrame:
    base = rb._add_rank_columns(rb._read_scores(with_labels=True))
    needed = ["trade_date", "stock_code", *FEATURES]
    quoted = ", ".join(f'"{c}"' for c in needed)
    with duckdb.connect(str(FEATURE_DB), read_only=True) as con:
        features = con.execute(
            f"""
            SELECT {quoted}
            FROM {FEATURE_TABLE}
            WHERE trade_date >= ?
              AND stock_code NOT LIKE '%.BJ'
            """,
            [rb.START_DATE],
        ).fetchdf()
    frame = base[["trade_date", "stock_code", "pred_10d_rank", LABEL_COL]].merge(
        features,
        on=["trade_date", "stock_code"],
        how="inner",
    )
    return frame[frame[LABEL_COL].notna()].copy()


def _condition_score(frame: pd.DataFrame, feature: str, threshold: float, bonus: float) -> pd.Series:
    rank = frame.groupby("trade_date")[feature].rank(method="average", pct=True)
    condition = rank <= threshold if threshold < 0.5 else rank >= threshold
    raw = frame["pred_10d_rank"].astype("float64") + bonus * condition.astype("float64")
    return raw.groupby(frame["trade_date"]).rank(method="average", pct=True)


def _daily_eval(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return rb._daily_eval(frame, score_col, LABEL_COL)


def _metric_blocks(daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "full": rb._metrics(daily),
        "train": rb._metrics(daily[daily["trade_date"] <= TRAIN_END]),
        "holdout": rb._metrics(daily[daily["trade_date"] >= HOLDOUT_START]),
        "recent20": rb._metrics(daily.tail(20)),
        "recent63": rb._metrics(daily.tail(63)),
        "recent126": rb._metrics(daily.tail(126)),
        "annual_stability": rb._period_stability(daily, "year"),
        "monthly_stability": rb._period_stability(daily, "month"),
        "eval_min_trade_date": str(daily["trade_date"].min()) if len(daily) else None,
        "eval_max_trade_date": str(daily["trade_date"].max()) if len(daily) else None,
        "eval_trade_days": int(len(daily)),
    }


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], block: str) -> dict[str, float | None]:
    return {
        key: rb._to_float((candidate[block].get(key) or 0.0) - (baseline[block].get(key) or 0.0))
        for key in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
    }


def _gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    full = _delta(candidate, baseline, "full")
    holdout = _delta(candidate, baseline, "holdout")
    recent63 = _delta(candidate, baseline, "recent63")
    recent20 = _delta(candidate, baseline, "recent20")
    if (full["rank_ic"] or 0.0) < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    for key in ["top1", "top3", "top5", "top10", "top20"]:
        if (full[key] or 0.0) <= 0.0:
            reasons.append(f"full_{key}_delta_non_positive")
    if (holdout["top5"] or 0.0) <= 0.0:
        reasons.append("holdout_top5_delta_non_positive")
    if (recent63["top5"] or 0.0) <= 0.0:
        reasons.append("recent63_top5_delta_non_positive")
    if (recent20["top5"] or 0.0) <= 0.0:
        reasons.append("recent20_top5_delta_non_positive")
    return not reasons, reasons


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = _load()
    baseline_frame = frame[["trade_date", "stock_code", LABEL_COL, "pred_10d_rank"]].copy()
    baseline_frame["score"] = baseline_frame["pred_10d_rank"]
    baseline = _metric_blocks(_daily_eval(baseline_frame, "score"))
    train_frame = frame[frame["trade_date"] <= TRAIN_END].copy()

    scan_rows: list[dict[str, Any]] = []
    best: tuple[float, str, float, float] | None = None
    for feature in FEATURES:
        for threshold in THRESHOLDS:
            for bonus in BONUSES:
                local = train_frame[["trade_date", LABEL_COL]].copy()
                local["score"] = _condition_score(train_frame, feature, threshold, bonus)
                local["rank_desc"] = local.groupby("trade_date")["score"].rank(ascending=False, method="first")
                local["base_rank_desc"] = train_frame.groupby("trade_date")["pred_10d_rank"].rank(
                    ascending=False,
                    method="first",
                )
                cand = local[local["rank_desc"] <= 5].groupby("trade_date")[LABEL_COL].mean().mean()
                base = local[local["base_rank_desc"] <= 5].groupby("trade_date")[LABEL_COL].mean().mean()
                delta = float(cand - base)
                scan_rows.append(
                    {
                        "feature": feature,
                        "threshold": threshold,
                        "side": "low" if threshold < 0.5 else "high",
                        "bonus": bonus,
                        "train_top5_delta": delta,
                    }
                )
                if best is None or delta > best[0]:
                    best = (delta, feature, threshold, bonus)

    assert best is not None
    _, feature, threshold, bonus = best
    candidate_frame = frame[["trade_date", "stock_code", LABEL_COL]].copy()
    candidate_frame["score"] = _condition_score(frame, feature, threshold, bonus)
    candidate = _metric_blocks(_daily_eval(candidate_frame, "score"))
    passed, failures = _gate(candidate, baseline)
    summary = {
        "label_key": LABEL_KEY,
        "selection_rule": "Only train-window Top5 delta over 20220606-20251231 is used for condition selection.",
        "selected": {
            "feature": feature,
            "threshold": threshold,
            "side": "low" if threshold < 0.5 else "high",
            "bonus": bonus,
            "train_top5_delta": best[0],
        },
        "deltas": {
            "full": _delta(candidate, baseline, "full"),
            "train": _delta(candidate, baseline, "train"),
            "holdout": _delta(candidate, baseline, "holdout"),
            "recent63": _delta(candidate, baseline, "recent63"),
            "recent20": _delta(candidate, baseline, "recent20"),
        },
        "candidate_metrics": candidate,
        "baseline_metrics": baseline,
        "gate": {"passed": passed, "failed_reasons": failures},
    }
    scan_csv = REPORT_DIR / "size_liquidity_condition_10d_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8-sig")
    report = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_10d_size_liquidity_condition_fast",
        "feature_db": str(FEATURE_DB),
        "feature_table": FEATURE_TABLE,
        "train_window": {"start": "20220606", "end": TRAIN_END},
        "holdout_window": {"start": HOLDOUT_START, "end": "mature_label_cutoff"},
        "summary": summary,
        "scan_csv": str(scan_csv),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_prediction_asset_write": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "size_liquidity_condition_10d_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 10D 市值/成交额条件化评分快速扫描",
        "",
        "## 结论",
        "",
        "本轮只用训练窗口 Top5 Δ 选择条件；holdout / recent 只用于验证。",
        "",
        f"- 选中条件：`{feature}` {'低分位' if threshold < 0.5 else '高分位'} `{threshold}`，bonus `{bonus}`",
        f"- 是否过门：`{passed}`",
        f"- Full Top5 Δ：`{summary['deltas']['full']['top5']:+.6f}`",
        f"- Holdout Top5 Δ：`{summary['deltas']['holdout']['top5']:+.6f}`",
        f"- Recent63 Top5 Δ：`{summary['deltas']['recent63']['top5']:+.6f}`",
        f"- Recent20 Top5 Δ：`{summary['deltas']['recent20']['top5']:+.6f}`",
        f"- 失败原因：`{'; '.join(failures) if failures else '-'}`",
        "",
        "## 边界",
        "",
        "- research-only。",
        "- 未训练模型。",
        "- 未写预测资产。",
        "- 未修改 formal manifest。",
        "- 未生成 `approved_for_l5`。",
        "- 未生成信号，未跑回测。",
        "",
        "## 证据",
        "",
        f"- 扫描明细：`{scan_csv}`",
        f"- 报告 JSON：`{report_json}`",
    ]
    (REPORT_DIR / "size_liquidity_condition_10d_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "report_json": str(report_json), "scan_csv": str(scan_csv)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
