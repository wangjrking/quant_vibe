from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_formal_l4_independent_alpha_review_20260721"
POOL_DB = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_smooth_frequency_20260719" / "current_formal_l4_rank_pool.duckdb"
LABEL_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_label_current.duckdb"
ROUND_TRIP_COST = 0.0066
TOP_NS = [1, 3, 5]

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "label": payload["label"],
        "manifest_path": str(path),
        "db_path": str((path.parent / payload["db_path"]).resolve()),
        "table": payload["table"],
        "source_type": payload["source_type"],
        "approval_status": payload["approval_status"],
        "max_trade_date": payload["max_trade_date"],
        "pred_prob_sha256": payload.get("pred_prob_sha256"),
    }


def period_slices(dates: list[str]) -> dict[str, list[str]]:
    return {
        "2022H2": [d for d in dates if "20220606" <= d <= "20221230"],
        "2023": [d for d in dates if "20230101" <= d <= "20231231"],
        "2024": [d for d in dates if "20240101" <= d <= "20241231"],
        "2025": [d for d in dates if "20250101" <= d <= "20251231"],
        "2026YTD": [d for d in dates if "20260101" <= d <= "20260616"],
        "recent120": dates[-120:] if len(dates) >= 120 else dates[:],
        "recent60": dates[-60:] if len(dates) >= 60 else dates[:],
    }


def summarize_periods(daily: pd.DataFrame) -> dict[str, dict]:
    dates = daily["trade_date"].astype(str).tolist()
    slices = period_slices(dates)
    out: dict[str, dict] = {}
    for name, period_dates in slices.items():
        part = daily[daily["trade_date"].isin(period_dates)].copy()
        if part.empty:
            continue
        record = {"trade_days": int(len(part))}
        for topn in TOP_NS:
            record[f"gross_top{topn}"] = float(part[f"gross_top{topn}"].mean())
            record[f"net_top{topn}"] = float(part[f"net_top{topn}"].mean())
        out[name] = record
    return out


def build_daily_metrics(con: duckdb.DuckDBPyConnection, horizon: str) -> pd.DataFrame:
    label_col = f"executable_{horizon}_open_return"
    score_col = f"rank_{horizon}"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW eval_{horizon} AS
        SELECT
            p.trade_date,
            p.stock_code,
            p.{score_col} AS model_rank,
            l.{label_col} AS realized
        FROM pool.current_formal_l4_rank_pool p
        JOIN labels.prod_l3_prediction_label_parts_current l
          ON p.trade_date = l.trade_date
         AND p.stock_code = l.stock_code
        WHERE l.{label_col} IS NOT NULL
        """
    )
    topn = con.execute(
        f"""
        WITH ranked AS (
            SELECT
                trade_date,
                stock_code,
                model_rank,
                realized,
                row_number() OVER (PARTITION BY trade_date ORDER BY model_rank DESC, stock_code) AS rn
            FROM eval_{horizon}
        )
        SELECT
            trade_date,
            COUNT(*) AS rows,
            AVG(realized) FILTER (WHERE rn <= 1) AS gross_top1,
            AVG(realized) FILTER (WHERE rn <= 3) AS gross_top3,
            AVG(realized) FILTER (WHERE rn <= 5) AS gross_top5
        FROM ranked
        GROUP BY trade_date
        ORDER BY trade_date
        """
    ).fetchdf()
    daily = topn
    for topn_val in TOP_NS:
        daily[f"net_top{topn_val}"] = daily[f"gross_top{topn_val}"] - ROUND_TRIP_COST
    return daily


def pairwise_overlap(con: duckdb.DuckDBPyConnection, left: str, right: str, topn: int) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH a AS (
            SELECT trade_date, stock_code
            FROM (
                SELECT
                    trade_date,
                    stock_code,
                    row_number() OVER (PARTITION BY trade_date ORDER BY model_rank DESC, stock_code) AS rn
                FROM eval_{left}
            )
            WHERE rn <= {topn}
        ),
        b AS (
            SELECT trade_date, stock_code
            FROM (
                SELECT
                    trade_date,
                    stock_code,
                    row_number() OVER (PARTITION BY trade_date ORDER BY model_rank DESC, stock_code) AS rn
                FROM eval_{right}
            )
            WHERE rn <= {topn}
        ),
        joined AS (
            SELECT a.trade_date, COUNT(*) AS overlap_count
            FROM a
            JOIN b USING (trade_date, stock_code)
            GROUP BY a.trade_date
        )
        SELECT
            trade_date,
            overlap_count / {float(topn)} AS overlap_ratio
        FROM joined
        ORDER BY trade_date
        """
    ).fetchdf()


def unique_top1_vs_10d(con: duckdb.DuckDBPyConnection, horizon: str) -> dict:
    if horizon == "10d":
        return {"different_days": 0, "mean_gross": None, "mean_net": None}
    result = con.execute(
        f"""
        WITH h AS (
            SELECT trade_date, stock_code, realized
            FROM (
                SELECT
                    trade_date,
                    stock_code,
                    realized,
                    row_number() OVER (PARTITION BY trade_date ORDER BY model_rank DESC, stock_code) AS rn
                FROM eval_{horizon}
            )
            WHERE rn = 1
        ),
        d10 AS (
            SELECT trade_date, stock_code
            FROM (
                SELECT
                    trade_date,
                    stock_code,
                    row_number() OVER (PARTITION BY trade_date ORDER BY model_rank DESC, stock_code) AS rn
                FROM eval_10d
            )
            WHERE rn = 1
        )
        SELECT
            COUNT(*) FILTER (WHERE h.stock_code <> d10.stock_code) AS different_days,
            AVG(h.realized) FILTER (WHERE h.stock_code <> d10.stock_code) AS mean_gross
        FROM h
        JOIN d10 USING (trade_date)
        """
    ).fetchone()
    gross = float(result[1]) if result[1] is not None else None
    return {
        "different_days": int(result[0]),
        "mean_gross": gross,
        "mean_net": None if gross is None else gross - ROUND_TRIP_COST,
    }


def verdict(horizon_rows: list[dict]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for row in horizon_rows:
        if row["horizon"] == "10d":
            continue
        if row["recent60_net_top3"] <= 0:
            reasons.append(f"{row['horizon']} recent60 Top3 成本后为负。")
        if row["recent120_net_top3"] <= 0:
            reasons.append(f"{row['horizon']} recent120 Top3 成本后为负。")
        if row["top5_overlap_vs_10d"] is not None and row["top5_overlap_vs_10d"] >= 0.6:
            reasons.append(f"{row['horizon']} 与 10D Top5 重合过高。")
        if row["top1_diff_net_vs_10d"] is not None and row["top1_diff_net_vs_10d"] <= 0:
            reasons.append(f"{row['horizon']} 在与 10D 不同 Top1 的日期里成本后无正边际。")
    if reasons:
        return "no_independent_alpha_in_current_formal", reasons
    return "possible_research_candidate_exists", reasons


def render_md(manifest_rows: list[dict], horizon_rows: list[dict], pair_rows: list[dict], status: str, reasons: list[str]) -> str:
    lines = [
        "# 当前 active formal L4 独立 Alpha 诊断",
        "",
        "## 当前结论",
        "",
    ]
    if status == "no_independent_alpha_in_current_formal":
        lines.append("当前 active formal L4 没有发现一个可直接交给策略侧做独立预注册验证的低相关、成本后稳定 Alpha 来源。10D 仍是主排序来源，1D/3D/5D 更像辅助或确认，而不是可单独成立的新前排 Alpha。")
    else:
        lines.append("当前 active formal L4 存在可继续研究的独立 Alpha 线索，但本报告不做 formal 或 production 建议。")
    lines.extend(["", "## 正式 manifest", ""])
    for row in manifest_rows:
        lines.append(f"- `{row['label']}`: `{row['manifest_path']}`")
    lines.extend(
        [
            "",
            "## 各期限摘要",
            "",
            "| horizon | full net top1 | full net top3 | full net top5 | recent120 net top3 | recent60 net top3 | top5 overlap vs 10D | top1 diff days vs 10D | top1 diff net vs 10D |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in horizon_rows:
        overlap = "" if row["top5_overlap_vs_10d"] is None else f"{row['top5_overlap_vs_10d']:.4f}"
        diff_net = "" if row["top1_diff_net_vs_10d"] is None else f"{row['top1_diff_net_vs_10d']:.4%}"
        lines.append(
            f"| {row['horizon']} | {row['full_net_top1']:.4%} | {row['full_net_top3']:.4%} | {row['full_net_top5']:.4%} | "
            f"{row['recent120_net_top3']:.4%} | {row['recent60_net_top3']:.4%} | {overlap} | "
            f"{row['top1_diff_days_vs_10d']} | {diff_net} |"
        )
    lines.extend(
        [
            "",
            "## 前排重合",
            "",
            "| pair | full top1 overlap | recent60 top1 overlap | full top3 overlap | recent60 top3 overlap | full top5 overlap | recent60 top5 overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pair_rows:
        lines.append(
            f"| {row['pair']} | {row['full_top1_overlap']:.4f} | {row['recent60_top1_overlap']:.4f} | "
            f"{row['full_top3_overlap']:.4f} | {row['recent60_top3_overlap']:.4f} | "
            f"{row['full_top5_overlap']:.4f} | {row['recent60_top5_overlap']:.4f} |"
        )
    lines.extend(["", "## 事实依据", ""])
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本报告只读，不训练、不调参、不改 formal manifest、不改 L4 数值表、不生成信号、不跑回测。",
            "- 日期冻结到 `20260616` 的成熟标签窗口；`20260722` 之后未见前向没有打开。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = [load_manifest(path) for path in MANIFESTS.values()]
    con = duckdb.connect()
    con.execute(f"ATTACH '{POOL_DB.as_posix()}' AS pool (READ_ONLY)")
    con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS labels (READ_ONLY)")
    horizon_rows: list[dict] = []
    pair_rows: list[dict] = []
    try:
        for horizon in MANIFESTS:
            daily = build_daily_metrics(con, horizon)
            periods = summarize_periods(daily)
            horizon_rows.append(
                {
                    "horizon": horizon,
                    "full_net_top1": float(daily["net_top1"].mean()),
                    "full_net_top3": float(daily["net_top3"].mean()),
                    "full_net_top5": float(daily["net_top5"].mean()),
                    "recent120_net_top3": float(periods["recent120"]["net_top3"]),
                    "recent60_net_top3": float(periods["recent60"]["net_top3"]),
                    "periods": periods,
                    "top1_diff_days_vs_10d": 0,
                    "top1_diff_net_vs_10d": None,
                    "top5_overlap_vs_10d": None,
                }
            )
        overlap_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
        for left, right in combinations(MANIFESTS.keys(), 2):
            record = {"pair": f"{left}-{right}"}
            for topn in TOP_NS:
                df = pairwise_overlap(con, left, right, topn)
                cache_key = tuple(sorted([left, right])) + (topn,)
                overlap_cache[cache_key] = df
                dates = df["trade_date"].astype(str).tolist()
                slices = period_slices(dates)
                record[f"full_top{topn}_overlap"] = float(df["overlap_ratio"].mean())
                record[f"recent60_top{topn}_overlap"] = float(df[df["trade_date"].isin(slices["recent60"])]["overlap_ratio"].mean())
            pair_rows.append(record)
        for row in horizon_rows:
            unique = unique_top1_vs_10d(con, row["horizon"])
            row["top1_diff_days_vs_10d"] = unique["different_days"]
            row["top1_diff_net_vs_10d"] = unique["mean_net"]
            if row["horizon"] == "10d":
                continue
            pair = tuple(sorted([row["horizon"], "10d"])) + (5,)
            row["top5_overlap_vs_10d"] = float(overlap_cache[pair]["overlap_ratio"].mean())
    finally:
        con.close()

    status, reasons = verdict(horizon_rows)
    payload = {
        "status": status,
        "analysis_date": "2026-07-21",
        "round_trip_cost": ROUND_TRIP_COST,
        "pool_db": str(POOL_DB),
        "label_db": str(LABEL_DB),
        "label_maturity_max_trade_date": "20260616",
        "manifest_rows": manifest_rows,
        "horizon_rows": horizon_rows,
        "pair_rows": pair_rows,
        "reasons": reasons,
        "boundary": {
            "no_training": True,
            "no_tuning": True,
            "no_formal_write": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "independent_alpha_review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in row.items() if k != "periods"} for row in horizon_rows]).to_csv(
        REPORT_DIR / "independent_alpha_horizon_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(pair_rows).to_csv(REPORT_DIR / "independent_alpha_pairwise_summary.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "独立Alpha诊断报告.md").write_text(
        render_md(manifest_rows, horizon_rows, pair_rows, status, reasons),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report_dir": str(REPORT_DIR)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
