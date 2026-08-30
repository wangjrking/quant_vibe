from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_four_year_top_consensus_holdout_20260714"
EXPERIMENT_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "experimental_assets"
    / "model-agent"
    / "l4_predictions"
    / "l4_top_consensus_holdout_20260714.duckdb"
)
LABEL_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"

START_DATE = "20220606"
SELECTION_END = "20251231"
HOLDOUT_START = "20260101"
TOP_K = [1, 3, 5, 10, 20]
RECENT_WINDOWS = [20, 63, 126]

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}
LABELS = {
    "1d": "executable_1d_open_return",
    "3d": "executable_3d_open_return",
    "5d": "executable_5d_open_return",
    "10d": "executable_10d_open_return",
}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _load_manifest(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(MAIN))
    from prediction_manifest import load_prediction_source_manifest

    return load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)


def _read_scores(with_labels: bool) -> pd.DataFrame:
    manifests = {key: _load_manifest(path) for key, path in MANIFESTS.items()}
    with duckdb.connect(database=":memory:") as con:
        con.execute(f"ATTACH '{Path(manifests['1d']['db_path']).as_posix()}' AS p1 (READ_ONLY)")
        con.execute(f"ATTACH '{Path(manifests['3d']['db_path']).as_posix()}' AS p3 (READ_ONLY)")
        con.execute(f"ATTACH '{Path(manifests['5d']['db_path']).as_posix()}' AS p5 (READ_ONLY)")
        con.execute(f"ATTACH '{Path(manifests['10d']['db_path']).as_posix()}' AS p10 (READ_ONLY)")
        if with_labels:
            con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS labels (READ_ONLY)")
            label_cols = ",\n                ".join(
                f"labels.{label} AS label_{key}" for key, label in LABELS.items()
            )
            label_join = f"JOIN labels.{LABEL_TABLE} labels USING (trade_date, stock_code)"
            label_select = f",\n                {label_cols}"
        else:
            label_join = ""
            label_select = ""

        return con.execute(
            f"""
            SELECT
                p1.trade_date,
                p1.stock_code,
                p1.pred_prob AS pred_1d,
                p3.pred_prob AS pred_3d,
                p5.pred_prob AS pred_5d,
                p10.pred_prob AS pred_10d
                {label_select}
            FROM p1.{manifests['1d']['table']} p1
            JOIN p3.{manifests['3d']['table']} p3 USING (trade_date, stock_code)
            JOIN p5.{manifests['5d']['table']} p5 USING (trade_date, stock_code)
            JOIN p10.{manifests['10d']['table']} p10 USING (trade_date, stock_code)
            {label_join}
            WHERE p1.trade_date >= ?
              AND p1.stock_code NOT LIKE '%.BJ'
              AND p1.pred_prob IS NOT NULL
              AND p3.pred_prob IS NOT NULL
              AND p5.pred_prob IS NOT NULL
              AND p10.pred_prob IS NOT NULL
            ORDER BY p1.trade_date, p1.stock_code
            """,
            [START_DATE],
        ).fetchdf()


def _add_rank_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ["pred_1d", "pred_3d", "pred_5d", "pred_10d"]:
        out[f"{column}_rank"] = out.groupby("trade_date")[column].rank(method="average", pct=True)
    return out


def _daily_eval(frame: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group[group[label_col].notna()]
        if len(group) < max(TOP_K):
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
        row: dict[str, Any] = {
            "trade_date": str(trade_date),
            "rows": int(len(ordered)),
            "rank_ic": _to_float(ordered[score_col].corr(ordered[label_col], method="spearman")),
        }
        for top_k in TOP_K:
            row[f"top{top_k}"] = _to_float(ordered.head(top_k)[label_col].mean())
        rows.append(row)
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    daily["year"] = daily["trade_date"].str.slice(0, 4)
    daily["month"] = daily["trade_date"].str.slice(0, 6)
    return daily


def _metrics(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return {"trade_days": 0}
    result: dict[str, Any] = {
        "trade_days": int(len(daily)),
        "rank_ic": _to_float(daily["rank_ic"].mean()),
        "rank_ic_positive_ratio": _to_float((daily["rank_ic"] > 0).mean()),
    }
    for top_k in TOP_K:
        result[f"top{top_k}"] = _to_float(daily[f"top{top_k}"].mean())
        result[f"top{top_k}_positive_ratio"] = _to_float((daily[f"top{top_k}"] > 0).mean())
    return result


def _period_stability(daily: pd.DataFrame, by: str) -> dict[str, Any]:
    metric_cols = ["rank_ic", *[f"top{k}" for k in TOP_K]]
    grouped = daily.groupby(by, sort=True)[metric_cols].mean().reset_index() if not daily.empty else pd.DataFrame()
    result: dict[str, Any] = {"periods": int(len(grouped))}
    for column in metric_cols:
        result[f"{column}_positive_period_ratio"] = (
            _to_float((grouped[column] > 0).mean()) if not grouped.empty else None
        )
        result[f"min_{column}"] = _to_float(grouped[column].min()) if not grouped.empty else None
    return result


def _four_year_gate(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    full = summary["full"]
    recent63 = summary["recent63"]
    recent20 = summary["recent20"]
    if summary["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    if (full.get("rank_ic") or 0.0) <= 0.0:
        reasons.append("full_rank_ic_non_positive")
    if (full.get("top5") or 0.0) <= 0.0:
        reasons.append("full_top5_non_positive")
    if (recent63.get("top5") or 0.0) <= 0.0:
        reasons.append("recent63_top5_non_positive")
    if (recent20.get("top5") or 0.0) <= 0.0:
        reasons.append("recent20_top5_non_positive")
    if (summary["monthly_stability"].get("top5_positive_period_ratio") or 0.0) < 0.5:
        reasons.append("monthly_top5_positive_ratio_below_0_5")
    if (summary["annual_stability"].get("top5_positive_period_ratio") or 0.0) < 0.75:
        reasons.append("annual_top5_positive_ratio_below_0_75")
    return not reasons, reasons


def _summarize_daily(daily: pd.DataFrame) -> dict[str, Any]:
    result = {
        "eval_min_trade_date": str(daily["trade_date"].min()) if not daily.empty else None,
        "eval_max_trade_date": str(daily["trade_date"].max()) if not daily.empty else None,
        "eval_trade_days": int(len(daily)),
        "full": _metrics(daily),
        "selection": _metrics(daily[daily["trade_date"] <= SELECTION_END]),
        "holdout": _metrics(daily[daily["trade_date"] >= HOLDOUT_START]),
        "recent20": _metrics(daily.tail(20)),
        "recent63": _metrics(daily.tail(63)),
        "recent126": _metrics(daily.tail(126)),
        "annual_stability": _period_stability(daily, "year"),
        "monthly_stability": _period_stability(daily, "month"),
    }
    passed, failures = _four_year_gate(result)
    result["four_year_gate"] = {"passed": passed, "failed_reasons": failures}
    return result


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    fields = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
    result: dict[str, Any] = {}
    for window in ["full", "selection", "holdout", "recent63", "recent20"]:
        for field in fields:
            result[f"{window}_{field}"] = _to_float(
                (candidate[window].get(field) or 0.0) - (baseline[window].get(field) or 0.0)
            )
    return result


def _candidate_score(
    frame: pd.DataFrame,
    *,
    target: str,
    aux: str,
    mode: str,
    alpha: float,
    threshold: float,
) -> pd.Series:
    target_rank = frame[f"pred_{target}_rank"]
    aux_rank = frame[f"pred_{aux}_rank"]
    mask = (target_rank >= threshold).astype("float64")
    if mode == "top_boost":
        score = target_rank + alpha * (aux_rank - 0.5) * mask
    elif mode == "top_penalty":
        score = target_rank - alpha * (0.5 - aux_rank).clip(lower=0.0) * mask
    elif mode == "consensus_min":
        score = (1.0 - alpha) * target_rank + alpha * pd.concat([target_rank, aux_rank], axis=1).min(axis=1)
    elif mode == "consensus_product":
        score = target_rank * (1.0 + alpha * aux_rank)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return score


def _candidate_grid(target: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {"target": target, "aux": target, "mode": "baseline", "alpha": 0.0, "threshold": 0.0}
    ]
    aux_by_target = {
        "3d": ["5d"],
        "5d": ["1d", "3d"],
        "10d": ["1d", "5d"],
        "1d": ["10d"],
    }
    for aux in aux_by_target.get(target, []):
        for mode in ["top_boost", "top_penalty"]:
            for alpha in [0.1, 0.2]:
                for threshold in [0.8, 0.9]:
                    candidates.append(
                        {"target": target, "aux": aux, "mode": mode, "alpha": alpha, "threshold": threshold}
                    )
        for mode in ["consensus_min"]:
            for alpha in [0.1, 0.2]:
                candidates.append(
                    {"target": target, "aux": aux, "mode": mode, "alpha": alpha, "threshold": 0.0}
                )
        candidates.append(
            {"target": target, "aux": aux, "mode": "consensus_product", "alpha": 0.1, "threshold": 0.0}
        )
    return candidates


def _evaluate_candidate(base: pd.DataFrame, label_key: str, spec: dict[str, Any]) -> dict[str, Any]:
    label_col = f"label_{label_key}"
    frame = base[base[label_col].notna()].copy()
    if spec["mode"] == "baseline":
        frame["candidate_score"] = frame[f"pred_{label_key}_rank"]
    else:
        frame["candidate_score"] = _candidate_score(
            frame,
            target=str(spec["target"]),
            aux=str(spec["aux"]),
            mode=str(spec["mode"]),
            alpha=float(spec["alpha"]),
            threshold=float(spec["threshold"]),
        )
    daily = _daily_eval(frame, "candidate_score", label_col)
    summary = _summarize_daily(daily)
    summary["spec"] = spec
    return summary


def _selection_objective(summary: dict[str, Any]) -> float:
    selection = summary["selection"]
    return (
        (selection.get("top5") or -1.0)
        + 0.4 * (selection.get("top3") or -1.0)
        + 0.3 * (selection.get("top10") or -1.0)
        + 0.05 * (selection.get("rank_ic") or -1.0)
    )


def _improvement_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    delta = _delta(candidate, baseline)
    if not candidate["four_year_gate"]["passed"]:
        reasons.append("candidate_four_year_gate_failed")
    if (delta.get("full_rank_ic") or 0.0) < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    for field in [
        "full_top5",
        "selection_top5",
        "holdout_top5",
        "recent63_top5",
        "recent20_top5",
    ]:
        if (delta.get(field) or 0.0) <= 0.0:
            reasons.append(f"{field}_delta_non_positive")
    return not reasons, reasons


def _flatten(label_key: str, item: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    delta = _delta(item, baseline)
    passed, failures = _improvement_gate(item, baseline)
    spec = item["spec"]
    return {
        "label_key": label_key,
        "target": spec["target"],
        "aux": spec["aux"],
        "mode": spec["mode"],
        "alpha": spec["alpha"],
        "threshold": spec["threshold"],
        "selection_objective": _selection_objective(item),
        "four_year_gate_passed": item["four_year_gate"]["passed"],
        "four_year_gate_failures": ";".join(item["four_year_gate"]["failed_reasons"]),
        "improvement_gate_passed": passed,
        "improvement_gate_failures": ";".join(failures),
        **{f"delta_{key}": value for key, value in delta.items()},
        "full_rank_ic": item["full"].get("rank_ic"),
        "full_top5": item["full"].get("top5"),
        "holdout_top5": item["holdout"].get("top5"),
        "recent63_top5": item["recent63"].get("top5"),
        "recent20_top5": item["recent20"].get("top5"),
    }


def _candidate_formula_text(spec: dict[str, Any]) -> str:
    if spec["mode"] == "baseline":
        return f"cross_section_percent_rank(active_formal_{spec['target']}.pred_prob)"
    return (
        f"mode={spec['mode']}; target={spec['target']}; aux={spec['aux']}; "
        f"alpha={spec['alpha']}; threshold={spec['threshold']}"
    )


def _materialize(full_base: pd.DataFrame, label_key: str, selected: dict[str, Any]) -> dict[str, Any]:
    spec = selected["spec"]
    table = f"stock_predict_data_model_agent_four_year_top_consensus_20260714_executable_{label_key}_open_return_research"
    out = full_base[["trade_date", "stock_code"]].copy()
    if spec["mode"] == "baseline":
        raw_score = full_base[f"pred_{label_key}_rank"]
    else:
        raw_score = _candidate_score(
            full_base,
            target=str(spec["target"]),
            aux=str(spec["aux"]),
            mode=str(spec["mode"]),
            alpha=float(spec["alpha"]),
            threshold=float(spec["threshold"]),
        )
    out["pred_prob"] = raw_score.groupby(full_base["trade_date"]).rank(method="average", pct=True)
    out["score_source"] = "top_consensus_holdout_20260714_research"
    EXPERIMENT_DB.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(EXPERIMENT_DB)) as con:
        con.register("candidate_frame", out)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM candidate_frame")
        latest_date = con.execute(f"SELECT max(trade_date) FROM {table}").fetchone()[0]
        quality = con.execute(
            f"""
            SELECT
                count(*) AS row_count,
                min(trade_date) AS min_trade_date,
                max(trade_date) AS max_trade_date,
                count(distinct trade_date) AS trade_days,
                count(distinct stock_code) AS stock_count,
                sum(case when pred_prob is null then 1 else 0 end) AS null_pred_prob,
                sum(case when stock_code like '%.BJ' then 1 else 0 end) AS bj_rows,
                (
                    SELECT count(*)
                    FROM (
                        SELECT trade_date, stock_code, count(*) c
                        FROM {table}
                        GROUP BY 1,2
                        HAVING c > 1
                    )
                ) AS duplicate_key_groups,
                sum(case when trade_date = ? then 1 else 0 end) AS latest_day_rows,
                count(distinct case when trade_date = ? then stock_code end) AS latest_day_stocks
            FROM {table}
            """,
            [latest_date, latest_date],
        ).fetchdf().iloc[0].to_dict()
        digest = hashlib.sha256()
        for trade_date, stock_code, pred_prob in con.execute(
            f"SELECT trade_date, stock_code, pred_prob FROM {table} ORDER BY trade_date, stock_code"
        ).fetchall():
            digest.update(f"{trade_date}|{stock_code}|{float(pred_prob):.17g}\n".encode("utf-8"))
    return {
        "db_path": str(EXPERIMENT_DB),
        "table": table,
        "quality": {key: _to_float(value) for key, value in quality.items()},
        "pred_prob_sha256": digest.hexdigest(),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    requested_labels = os.environ.get("MODEL_RESEARCH_LABEL_KEYS")
    label_keys = (
        [item.strip() for item in requested_labels.split(",") if item.strip()]
        if requested_labels
        else ["3d", "5d", "10d"]
    )
    invalid = sorted(set(label_keys) - set(LABELS))
    if invalid:
        raise ValueError(f"unsupported label keys: {invalid}")
    base = _add_rank_columns(_read_scores(with_labels=True))
    full_base = _add_rank_columns(_read_scores(with_labels=False))

    scan_rows: list[dict[str, Any]] = []
    label_reports: dict[str, Any] = {}
    for label_key in label_keys:
        baseline = _evaluate_candidate(
            base,
            label_key,
            {"target": label_key, "aux": label_key, "mode": "baseline", "alpha": 0.0, "threshold": 0.0},
        )
        candidates = [_evaluate_candidate(base, label_key, spec) for spec in _candidate_grid(label_key)]
        for candidate in candidates:
            scan_rows.append(_flatten(label_key, candidate, baseline))

        eligible = []
        for candidate in candidates:
            pass_gate, failures = _improvement_gate(candidate, baseline)
            if pass_gate:
                eligible.append(candidate)
        selected_by_train = sorted(candidates, key=_selection_objective, reverse=True)[0]
        selected_delta = _delta(selected_by_train, baseline)
        selected_passed, selected_failures = _improvement_gate(selected_by_train, baseline)

        promoted = None
        if selected_passed:
            promoted = _materialize(full_base, label_key, selected_by_train)

        best_eligible = sorted(eligible, key=_selection_objective, reverse=True)[0] if eligible else None
        label_reports[label_key] = {
            "label": LABELS[label_key],
            "baseline": baseline,
            "selected_by_selection_window": selected_by_train,
            "selected_delta_vs_baseline": selected_delta,
            "selected_improvement_gate": {
                "passed": selected_passed,
                "failed_reasons": selected_failures,
            },
            "eligible_candidate_count": len(eligible),
            "best_eligible_candidate": best_eligible,
            "materialized_candidate": promoted,
        }

    suffix = "_".join(label_keys)
    scan_csv = REPORT_DIR / f"top_consensus_holdout_{suffix}_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "scope": "research_only_four_year_top_consensus_holdout",
        "selection_window": {"min_trade_date": START_DATE, "max_trade_date": SELECTION_END},
        "holdout_window": {"min_trade_date": HOLDOUT_START, "max_trade_date": "label_maturity_by_horizon"},
        "candidate_db": str(EXPERIMENT_DB),
        "labels": label_reports,
        "scan_csv": str(scan_csv),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
            "label_maturity_respected_for_evaluation": True,
        },
    }
    report_json = REPORT_DIR / f"top_consensus_holdout_{suffix}_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 四年观察期 Top 共识评分 holdout 研究",
        "",
        "## 边界",
        "",
        "- 本轮仅为 research-only 评分改造实验。",
        "- 未训练模型、未调参生产模型、未修改 formal manifest、未生成交易信号、未运行回测。",
        "- 选择窗口为 20220606-20251231，holdout 为 20260101 到各标签成熟截止日。",
        "",
        "## 结论摘要",
        "",
        "| 标签 | 选择窗口选中公式 | 是否通过 holdout 改善门 | 合格候选数 | Full Top5 增量 | Holdout Top5 增量 | Recent63 Top5 增量 | Recent20 Top5 增量 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label_key, item in label_reports.items():
        delta = item["selected_delta_vs_baseline"]
        gate = item["selected_improvement_gate"]
        lines.append(
            "| {label} | `{formula}` | {passed} | {count} | {full_top5:.6f} | {holdout_top5:.6f} | {r63:.6f} | {r20:.6f} |".format(
                label=label_key,
                formula=_candidate_formula_text(item["selected_by_selection_window"]["spec"]),
                passed="通过" if gate["passed"] else "未通过",
                count=item["eligible_candidate_count"],
                full_top5=delta["full_top5"] or 0.0,
                holdout_top5=delta["holdout_top5"] or 0.0,
                r63=delta["recent63_top5"] or 0.0,
                r20=delta["recent20_top5"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## 证据",
            "",
            f"- JSON：`{report_json}`",
            f"- 扫描明细：`{scan_csv}`",
            f"- 实验库：`{EXPERIMENT_DB}`",
        ]
    )
    report_md = REPORT_DIR / f"top_consensus_holdout_{suffix}_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": "ok", "report_json": str(report_json), "scan_csv": str(scan_csv)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
