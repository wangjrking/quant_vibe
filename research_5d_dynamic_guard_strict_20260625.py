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
SOURCE_REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_dynamic_guard_research_20260625"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_dynamic_guard_strict_research_20260625"
SOURCE_TABLE = "stock_predict_data_model_agent_5d_dynamic_guard_20260625_executable_5d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_5d_dynamic_guard_strict_20260625_executable_5d_open_return_research"
LABEL = "executable_5d_open_return"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def top_set(group: pd.DataFrame, rank_col: str, n: int) -> set[str]:
    return set(group.nlargest(min(n, len(group)), rank_col)["stock_code"].astype(str))


def choose_strict_rule() -> dict:
    scan = pd.read_csv(SOURCE_REPORT_DIR / "dynamic_source_guard_scan_results.csv")
    sub = scan[
        (scan["active_ratio"] <= 0.25)
        & (scan["recent63_active_days"] >= 5)
        & (scan["recent63_top1_delta"] > 0)
        & (scan["recent63_top3_delta"] > 0)
        & (scan["recent63_top5_delta"] > 0)
        & (scan["recent63_top10_delta"] > 0)
        & (scan["recent63_rank_ic_delta"] >= -0.002)
        & (scan["recent20_top5_delta"] >= 0)
        & (scan["full_top5_delta"] >= -0.002)
        & (scan["full_rank_ic_delta"] >= -0.0015)
    ].copy()
    if sub.empty:
        raise RuntimeError("no strict 5D dynamic guard candidate passed the research gates")
    sub["strict_objective"] = (
        3.0 * sub["recent63_top1_delta"]
        + 2.0 * sub["recent63_top3_delta"]
        + 2.0 * sub["recent63_top5_delta"]
        + sub["recent20_top5_delta"]
        + 0.5 * sub["full_top5_delta"]
        + 0.2 * sub["recent63_rank_ic_delta"]
        - 0.01 * sub["active_ratio"]
    )
    best = sub.sort_values("strict_objective", ascending=False).iloc[0]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in best.to_dict().items()}


def build_strict_frame(rule: dict) -> tuple[pd.DataFrame, int]:
    with sqlite3.connect(MODEL_DB) as conn:
        frame = pd.read_sql_query(
            f"""
            select trade_date, stock_code, formal5, formal10, dyn10, guard3, formal1,
                   formal5_rank, formal10_rank, dyn10_rank, guard3_rank, formal1_rank
            from {quote(SOURCE_TABLE)}
            order by trade_date, stock_code
            """,
            conn,
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    overlaps = {}
    for trade_date, group in frame.groupby("trade_date", sort=True):
        overlaps[trade_date] = len(top_set(group, "formal5_rank", 10) & top_set(group, "guard3_rank", 10))
    frame["overlap10_formal5_guard3"] = frame["trade_date"].map(overlaps).astype(float)
    threshold = float(rule["threshold"])
    op = str(rule["op"])
    if op == "<=":
        frame["guard_active"] = frame["overlap10_formal5_guard3"] <= threshold
    elif op == ">=":
        frame["guard_active"] = frame["overlap10_formal5_guard3"] >= threshold
    else:
        raise ValueError(f"unsupported op: {op}")
    frame["pred_prob"] = np.where(frame["guard_active"], frame["dyn10_rank"], frame["formal5_rank"])
    frame["score_formula"] = f"if overlap10_formal5_guard3 {op} {threshold:.12g} then dyn10_rank else formal5_rank"
    active_days = int(frame.groupby("trade_date")["guard_active"].first().sum())
    return frame, active_days


def write_table(frame: pd.DataFrame, active_days: int) -> dict:
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
        "overlap10_formal5_guard3",
        "guard_active",
        "score_formula",
    ]
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        frame[keep].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
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
        "formula": str(frame["score_formula"].iloc[0]),
        "active_days_full": active_days,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def write_outputs(rule: dict, stats: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "model_track": "research",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "source_type": "sqlite_table",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db",
        "table": TARGET_TABLE,
        "label": LABEL,
        "generated_at": now_iso(),
        "decision": "strict_candidate",
        "formula": stats["formula"],
        "source_table": SOURCE_TABLE,
        "source_scan": str((SOURCE_REPORT_DIR / "dynamic_source_guard_scan_results.csv").as_posix()),
        "best_scan": rule,
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
    (REPORT_DIR / "strict_dynamic_guard_research_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([rule]).to_csv(REPORT_DIR / "strict_dynamic_guard_selected_rule.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 5D 严格动态守卫研究报告（20260625）",
        "",
        "## 当前结论",
        "",
        "本轮从 5D 动态守卫扫描结果中筛出一个更严格的 research-only 候选。相比上一版 5D dynamic guard 约 53% 触发，该候选只在约 17% 交易日切换到 10D dynamic rank。",
        "",
        "候选规则：",
        "",
        "```text",
        stats["formula"],
        "```",
        "",
        "该规则只使用同日 5D 与 3D 高分股票池重叠度，不使用未来标签；但阈值来自研究扫描，因此不能直接作为生产模型或 L5 输入。",
        "",
        "## 候选资产",
        "",
        f"- 候选表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
        "- 状态：`research_only_not_approved_for_l4_or_l5`",
        f"- 日期范围：`{stats['min_trade_date']}` 到 `{stats['max_trade_date']}`",
        f"- 总行数：`{stats['row_count']}`",
        f"- 交易日数：`{stats['trade_days']}`",
        f"- 最新日：`{stats['latest_days'][0][0]}`，行数 `{stats['latest_days'][0][1]}`，股票数 `{stats['latest_days'][0][2]}`",
        f"- 触发交易日数：`{stats['active_days_full']}`",
        f"- `pred_prob` 空值：`{stats['null_pred_prob']}`",
        f"- `(trade_date, stock_code)` 重复键组：`{stats['duplicate_key_groups']}`",
        "",
        "## 相对 formal 5D 的评价增量",
        "",
        f"- 全样本 RankIC：`{rule['full_rank_ic_delta']:.6f}`",
        f"- 全样本 Top1：`{rule['full_top1_delta']:.6f}`",
        f"- 全样本 Top5：`{rule['full_top5_delta']:.6f}`",
        f"- 近 63 日 RankIC：`{rule['recent63_rank_ic_delta']:.6f}`",
        f"- 近 63 日 Top1：`{rule['recent63_top1_delta']:.6f}`",
        f"- 近 63 日 Top3：`{rule['recent63_top3_delta']:.6f}`",
        f"- 近 63 日 Top5：`{rule['recent63_top5_delta']:.6f}`",
        f"- 近 63 日 Top10：`{rule['recent63_top10_delta']:.6f}`",
        "",
        "## 证据路径",
        "",
        f"- 研究 manifest：`{(REPORT_DIR / 'strict_dynamic_guard_research_manifest.json').as_posix()}`",
        f"- 选中规则：`{(REPORT_DIR / 'strict_dynamic_guard_selected_rule.csv').as_posix()}`",
        f"- 原扫描明细：`{(SOURCE_REPORT_DIR / 'dynamic_source_guard_scan_results.csv').as_posix()}`",
        "",
        "## 边界说明",
        "",
        "- 未训练模型。",
        "- 未调参训练参数。",
        "- 未修改 production manifest。",
        "- 未写入 formal L4 表。",
        "- 未修改 `production_tasks.json`。",
        "- 未生成交易信号。",
        "- 未运行策略回测。",
    ]
    (REPORT_DIR / "strict_dynamic_guard_research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rule = choose_strict_rule()
    frame, active_days = build_strict_frame(rule)
    stats = write_table(frame, active_days)
    write_outputs(rule, stats)
    print(json.dumps({"rule": rule, "asset_stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
