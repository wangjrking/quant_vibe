from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_dynamic_guard_combo_research_20260625"

SOURCE_TABLE = "stock_predict_data_model_agent_5d_dynamic_guard_20260625_executable_5d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_5d_dynamic_guard_combo_20260625_executable_5d_open_return_research"
LABEL = "executable_5d_open_return"
BENCHMARK_ASSET = "research_5d_dynamic_guard"
BENCHMARK_STABILITY_OBJECTIVE = 0.15881603802773198
BENCHMARK_ACTIVE_DAYS = 267


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def top_set(group: pd.DataFrame, rank_col: str, n: int) -> set[str]:
    return set(group.nlargest(min(n, len(group)), rank_col)["stock_code"].astype(str))


def load_frame() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB) as conn:
        frame = pd.read_sql_query(
            f"""
            select trade_date, stock_code, formal5, formal10, dyn10, guard3, formal1,
                   formal5_rank, formal10_rank, dyn10_rank, guard3_rank, formal1_rank,
                   overlap50_formal5_formal10
            from {quote(SOURCE_TABLE)}
            order by trade_date, stock_code
            """,
            conn,
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    feature_rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        formal5_top10 = top_set(group, "formal5_rank", 10)
        formal5_top20 = top_set(group, "formal5_rank", 20)
        formal5_top50 = top_set(group, "formal5_rank", 50)
        feature_rows.append(
            {
                "trade_date": trade_date,
                "overlap10_formal5_guard3": len(formal5_top10 & top_set(group, "guard3_rank", 10)),
                "overlap20_formal5_guard3": len(formal5_top20 & top_set(group, "guard3_rank", 20)),
                "overlap50_formal5_guard3": len(formal5_top50 & top_set(group, "guard3_rank", 50)),
                "overlap10_formal5_formal10": len(formal5_top10 & top_set(group, "formal10_rank", 10)),
                "overlap20_formal5_formal10": len(formal5_top20 & top_set(group, "formal10_rank", 20)),
                "overlap50_formal5_formal10_calc": len(formal5_top50 & top_set(group, "formal10_rank", 50)),
            }
        )
    features = pd.DataFrame(feature_rows)
    out = frame.merge(features, on="trade_date", how="left", validate="many_to_one")
    out["overlap50_formal5_formal10"] = out["overlap50_formal5_formal10"].fillna(out["overlap50_formal5_formal10_calc"])
    return out


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in LABEL_DIR.glob("*.parquet"):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    label = pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])
    return label


def top_mean(group: pd.DataFrame, score_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), score_col)[LABEL].mean()) if len(group) else np.nan


def bottom_mean(group: pd.DataFrame, score_col: str, n: int) -> float:
    return float(group.nsmallest(min(n, len(group)), score_col)[LABEL].mean()) if len(group) else np.nan


def daily_metrics(eval_frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for trade_date, group in eval_frame.groupby("trade_date", sort=True):
        rank = group[score_col].rank(method="average", pct=True)
        label_rank = group[LABEL].rank(method="average", pct=True)
        row = {
            "trade_date": trade_date,
            "rank_ic": float(rank.corr(label_rank)),
            "pearson_ic": float(rank.corr(group[LABEL])),
        }
        tmp = group.copy()
        tmp["_rank"] = rank
        for n in [1, 3, 5, 10, 20, 50]:
            row[f"top{n}"] = top_mean(tmp, "_rank", n)
        row["top_bottom"] = top_mean(tmp, "_rank", 50) - bottom_mean(tmp, "_rank", 50)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame, dates: list[str], n: int | None) -> dict:
    use = set(dates if n is None else dates[-n:])
    sub = daily[daily["trade_date"].isin(use)]
    return {
        col: float(sub[col].mean())
        for col in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]
    }


def period_score(daily_delta: pd.DataFrame) -> dict:
    out: dict[str, float | int] = {}
    periods = {
        "2024H2": ("20240604", "20241231"),
        "2025H1": ("20250101", "20250630"),
        "2025H2": ("20250701", "20251231"),
        "2026YTD": ("20260101", "99999999"),
    }
    top5s = []
    rankics = []
    top1s = []
    for name, (lo, hi) in periods.items():
        sub = daily_delta[(daily_delta["trade_date"] >= lo) & (daily_delta["trade_date"] <= hi)]
        top5 = float(sub["top5_delta"].mean())
        rank_ic = float(sub["rank_ic_delta"].mean())
        top1 = float(sub["top1_delta"].mean())
        out[f"{name}_top5_delta"] = top5
        out[f"{name}_rank_ic_delta"] = rank_ic
        out[f"{name}_top1_delta"] = top1
        top5s.append(top5)
        rankics.append(rank_ic)
        top1s.append(top1)
    out["positive_top5_periods"] = int(sum(v > 0 for v in top5s))
    out["positive_top1_periods"] = int(sum(v > 0 for v in top1s))
    out["min_period_top5_delta"] = float(min(top5s))
    out["min_period_rank_ic_delta"] = float(min(rankics))
    out["avg_period_top5_delta"] = float(np.mean(top5s))
    out["avg_period_rank_ic_delta"] = float(np.mean(rankics))
    return out


def candidate_conditions(day_features: pd.DataFrame) -> list[dict]:
    conditions = []
    specs = [
        ("overlap50_formal5_formal10", ">=", [28, 30, 32, 35]),
        ("overlap50_formal5_guard3", ">=", [30, 32, 35, 38]),
        ("overlap20_formal5_formal10", ">=", [10, 12, 14, 16]),
        ("overlap20_formal5_guard3", ">=", [14, 16, 17, 18]),
        ("overlap10_formal5_guard3", "<=", [3, 4, 5]),
        ("overlap10_formal5_guard3", ">=", [8, 9, 10]),
    ]
    for feature, op, thresholds in specs:
        for threshold in thresholds:
            conditions.append({"name": f"{feature} {op} {threshold}", "feature": feature, "op": op, "threshold": float(threshold)})
    return conditions


def apply_condition(day_features: pd.DataFrame, cond: dict) -> pd.Series:
    vals = day_features[cond["feature"]]
    if cond["op"] == ">=":
        return vals >= cond["threshold"]
    if cond["op"] == "<=":
        return vals <= cond["threshold"]
    raise ValueError(cond["op"])


def scan(frame: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_features = frame.groupby("trade_date", as_index=False).first()[
        [
            "trade_date",
            "overlap50_formal5_formal10",
            "overlap50_formal5_guard3",
            "overlap20_formal5_formal10",
            "overlap20_formal5_guard3",
            "overlap10_formal5_guard3",
        ]
    ]
    conditions = candidate_conditions(day_features)
    joined_base = frame.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    dates = sorted(joined_base["trade_date"].unique().tolist())
    base_daily = daily_metrics(joined_base.assign(candidate_score=joined_base["formal5_rank"]), "candidate_score")
    dyn10_daily = daily_metrics(joined_base.assign(candidate_score=joined_base["dyn10_rank"]), "candidate_score")
    base_daily_idx = base_daily.set_index("trade_date")
    dyn10_daily_idx = dyn10_daily.set_index("trade_date")
    rows = []
    daily_records = []
    for i, left in enumerate(conditions):
        left_active = apply_condition(day_features, left)
        base_rule_defs = [(left["name"], left_active)]
        for right in conditions[i + 1 :]:
            right_active = apply_condition(day_features, right)
            base_rule_defs.append((f"({left['name']}) OR ({right['name']})", left_active | right_active))
            base_rule_defs.append((f"({left['name']}) AND ({right['name']})", left_active & right_active))
        for rule_name, active in base_rule_defs:
            active_dates = set(day_features.loc[active, "trade_date"].astype(str))
            if len(active_dates) < 10 or len(active_dates) > int(0.55 * len(day_features)):
                continue
            cand_daily = base_daily_idx.copy()
            active_eval_dates = [d for d in active_dates if d in dyn10_daily_idx.index]
            cand_daily.loc[active_eval_dates, :] = dyn10_daily_idx.loc[active_eval_dates, cand_daily.columns]
            cand_daily = cand_daily.reset_index()
            merged = cand_daily.merge(base_daily, on="trade_date", suffixes=("", "_base"))
            for metric in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
                merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_base"]
            full = summarize(cand_daily, dates, None)
            base_full = summarize(base_daily, dates, None)
            r63 = summarize(cand_daily, dates, 63)
            b63 = summarize(base_daily, dates, 63)
            r20 = summarize(cand_daily, dates, 20)
            b20 = summarize(base_daily, dates, 20)
            periods = period_score(merged)
            row = {
                "rule": rule_name,
                "active_days": len(active_dates),
                "active_ratio": len(active_dates) / len(day_features),
                "recent63_active_days": len(set(dates[-63:]) & active_dates),
                "recent20_active_days": len(set(dates[-20:]) & active_dates),
            }
            for prefix, cur, base in [("full", full, base_full), ("recent63", r63, b63), ("recent20", r20, b20)]:
                for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]:
                    row[f"{prefix}_{metric}_delta"] = cur[metric] - base[metric]
            row.update(periods)
            row["pass_basic"] = bool(
                row["recent63_active_days"] >= 5
                and row["recent63_top1_delta"] > 0
                and row["recent63_top5_delta"] > 0
                and row["recent63_top10_delta"] > 0
                and row["recent20_top5_delta"] >= 0
                and row["positive_top5_periods"] >= 3
                and row["full_top5_delta"] >= -0.001
                and row["full_rank_ic_delta"] >= -0.0015
            )
            row["stability_objective"] = (
                3.0 * row["recent63_top1_delta"]
                + 2.0 * row["recent63_top5_delta"]
                + row["recent63_top10_delta"]
                + row["recent20_top5_delta"]
                + 0.5 * row["avg_period_top5_delta"]
                + 0.01 * row["positive_top5_periods"]
                - 0.01 * row["active_ratio"]
                - 0.05 * max(0.0, -row["min_period_top5_delta"])
            )
            rows.append(row)
            if row["pass_basic"]:
                tmp = merged[["trade_date", "rank_ic_delta", "top1_delta", "top5_delta", "top10_delta"]].copy()
                tmp["rule"] = rule_name
                daily_records.append(tmp)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["pass_basic", "stability_objective"], ascending=[False, False]).reset_index(drop=True)
    daily_result = pd.concat(daily_records, ignore_index=True) if daily_records else pd.DataFrame()
    return result, daily_result


def write_candidate(frame: pd.DataFrame, best: pd.Series) -> dict:
    rule = str(best["rule"])
    day_features = frame.groupby("trade_date", as_index=False).first()[
        [
            "trade_date",
            "overlap50_formal5_formal10",
            "overlap50_formal5_guard3",
            "overlap20_formal5_formal10",
            "overlap20_formal5_guard3",
            "overlap10_formal5_guard3",
        ]
    ]
    # Evaluate the textual rule with a constrained expression namespace.
    expr = rule
    for col in [c for c in day_features.columns if c != "trade_date"]:
        expr = expr.replace(col, f"day_features[{col!r}]")
    expr = expr.replace(" AND ", " & ").replace(" OR ", " | ")
    active = eval(expr, {"__builtins__": {}}, {"day_features": day_features})
    active_dates = set(day_features.loc[active, "trade_date"].astype(str))
    out = frame.copy()
    out["guard_active"] = out["trade_date"].isin(active_dates)
    out["pred_prob"] = np.where(out["guard_active"], out["dyn10_rank"], out["formal5_rank"])
    out["score_formula"] = f"if {rule} then dyn10_rank else formal5_rank"
    keep = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "formal5",
        "formal10",
        "dyn10",
        "guard3",
        "formal1",
        "formal5_rank",
        "dyn10_rank",
        "guard3_rank",
        "overlap50_formal5_formal10",
        "overlap50_formal5_guard3",
        "overlap20_formal5_formal10",
        "overlap20_formal5_guard3",
        "overlap10_formal5_guard3",
        "guard_active",
        "score_formula",
    ]
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[keep].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        short = hashlib.sha1(TARGET_TABLE.encode("utf-8")).hexdigest()[:12]
        conn.execute(f"create index if not exists idx_{short}_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.execute(f"create index if not exists idx_{short}_date_pred on {quote(TARGET_TABLE)}(trade_date, pred_prob desc)")
        conn.commit()
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        latest = conn.execute(
            f"select trade_date, count(*), count(distinct stock_code) from {quote(TARGET_TABLE)} group by trade_date order by trade_date desc limit 10"
        ).fetchall()
    return {
        "table": TARGET_TABLE,
        "formula": out["score_formula"].iloc[0],
        "active_days_full": int(out.groupby("trade_date")["guard_active"].first().sum()),
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def write_outputs(scan_result: pd.DataFrame, daily_result: pd.DataFrame, stats: dict | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scan_path = REPORT_DIR / "combo_gate_scan_results.csv"
    daily_path = REPORT_DIR / "combo_gate_daily_deltas.csv"
    scan_result.to_csv(scan_path, index=False, encoding="utf-8-sig")
    daily_result.to_csv(daily_path, index=False, encoding="utf-8-sig")
    best = scan_result.iloc[0].to_dict() if not scan_result.empty else None
    recommended = bool(
        stats
        and best
        and best["stability_objective"] > BENCHMARK_STABILITY_OBJECTIVE
        and stats["active_days_full"] <= BENCHMARK_ACTIVE_DAYS
    )
    decision = "candidate_recommended" if recommended else ("rejected_no_improvement_over_existing_5d_dynamic_guard" if stats else "rejected")
    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset" if recommended else "l4_research_rejected_scan_artifact",
        "model_track": "research",
        "approval_status": "research_only_not_approved_for_l4_or_l5" if recommended else "research_rejected_no_improvement",
        "source_type": "sqlite_table" if stats else "scan_report_only",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db" if stats else None,
        "table": TARGET_TABLE if stats else None,
        "label": LABEL,
        "generated_at": now_iso(),
        "decision": decision,
        "recommended_for_next_research_or_strategy_validation": recommended,
        "benchmark": {
            "asset": BENCHMARK_ASSET,
            "stability_objective": BENCHMARK_STABILITY_OBJECTIVE,
            "active_days": BENCHMARK_ACTIVE_DAYS,
        },
        "best_scan": best,
        "asset_stats": stats,
        "promotion_requires_user_confirmation": True,
        "audit_required_before_formal": True,
        "boundaries": {
            "no_training": True,
            "no_tuning_training_parameters": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "combo_gate_research_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 5D 组合动态守卫研究报告（20260625）",
        "",
        "## 当前结论",
        "",
    ]
    if stats:
        lines.extend(
            [
                "本轮找到一个通过基础门槛的 5D 组合 gate research-only 候选，但它没有超过现有 `research_5d_dynamic_guard` 的稳定性基准，因此标记为不建议推进。",
                "",
                "候选规则：",
                "",
                "```text",
                stats["formula"],
                "```",
                "",
                f"- 候选表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
                f"- 日期范围：`{stats['min_trade_date']}` 到 `{stats['max_trade_date']}`",
                f"- 总行数：`{stats['row_count']}`",
                f"- 最新日：`{stats['latest_days'][0][0]}`，行数 `{stats['latest_days'][0][1]}`，股票数 `{stats['latest_days'][0][2]}`",
                f"- 触发交易日数：`{stats['active_days_full']}`",
                f"- `pred_prob` 空值：`{stats['null_pred_prob']}`",
                f"- 重复键组：`{stats['duplicate_key_groups']}`",
                f"- 是否建议推进：`{recommended}`",
                "",
                "## 评价摘要",
                "",
                f"- 近 63 日 Top1 增量：`{best['recent63_top1_delta']:.6f}`",
                f"- 近 63 日 Top5 增量：`{best['recent63_top5_delta']:.6f}`",
                f"- 近 63 日 Top10 增量：`{best['recent63_top10_delta']:.6f}`",
                f"- 全样本 Top5 增量：`{best['full_top5_delta']:.6f}`",
                f"- 正 Top5 自然阶段数：`{int(best['positive_top5_periods'])}/4`",
                f"- 最差自然阶段 Top5 增量：`{best['min_period_top5_delta']:.6f}`",
                f"- 本候选稳定性目标：`{best['stability_objective']:.6f}`",
                f"- 现有 5D dynamic guard 稳定性目标：`{BENCHMARK_STABILITY_OBJECTIVE:.6f}`",
            ]
        )
    else:
        lines.append("本轮未找到优于既有 5D dynamic guard 的组合 gate 候选，仅保留 rejected scan artifact。")
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 扫描明细：`{scan_path.as_posix()}`",
            f"- 日度增量：`{daily_path.as_posix()}`",
            f"- 研究 manifest：`{(REPORT_DIR / 'combo_gate_research_manifest.json').as_posix()}`",
            "",
            "## 边界说明",
            "",
            "- 未训练模型。",
            "- 未调参训练参数。",
            "- 未修改 production manifest。",
            "- 未写入 formal L4 表。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
        ]
    )
    (REPORT_DIR / "combo_gate_research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    frame = load_frame()
    labels = load_labels(str(frame["trade_date"].min()), str(frame["trade_date"].max()))
    scan_result, daily_result = scan(frame, labels)
    stats = None
    if not scan_result.empty and bool(scan_result.iloc[0]["pass_basic"]):
        stats = write_candidate(frame, scan_result.iloc[0])
    write_outputs(scan_result, daily_result, stats)
    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "scan_rows": int(len(scan_result)),
                "daily_rows": int(len(daily_result)),
                "best": scan_result.iloc[0].to_dict() if not scan_result.empty else None,
                "asset_stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
