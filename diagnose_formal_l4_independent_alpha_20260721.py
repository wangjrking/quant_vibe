from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_formal_l4_independent_alpha_review_20260721"
LABEL_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_label_current.duckdb"
ROUND_TRIP_COST = 0.0066

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}

TOP_NS = [1, 3, 5]
PERIOD_NAMES = ["2022H2", "2023", "2024", "2025", "2026YTD", "recent120", "recent60"]


def load_manifest(label: str) -> dict:
    path = MANIFESTS[label]
    payload = json.loads(path.read_text(encoding="utf-8"))
    db_path = (path.parent / payload["db_path"]).resolve()
    return {
        "label": payload["label"],
        "manifest_path": str(path),
        "db_path": str(db_path),
        "table": payload["table"],
        "source_type": payload["source_type"],
        "approval_status": payload["approval_status"],
        "max_trade_date": payload["max_trade_date"],
        "pred_prob_sha256": payload.get("pred_prob_sha256"),
    }


def period_slices(dates: list[str]) -> dict[str, list[str]]:
    result = {
        "2022H2": [d for d in dates if "20220606" <= d <= "20221230"],
        "2023": [d for d in dates if "20230101" <= d <= "20231231"],
        "2024": [d for d in dates if "20240101" <= d <= "20241231"],
        "2025": [d for d in dates if "20250101" <= d <= "20251231"],
        "2026YTD": [d for d in dates if "20260101" <= d <= "20260616"],
        "recent120": dates[-120:] if len(dates) >= 120 else dates[:],
        "recent60": dates[-60:] if len(dates) >= 60 else dates[:],
    }
    return result


def build_daily_metrics(con: duckdb.DuckDBPyConnection, horizon: str, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_col = meta["label"]
    pred_table = meta["table"]
    pred_alias = f"pred_{horizon}"
    view_name = f"eval_{horizon}"
    db_path = meta["db_path"].replace("'", "''")
    con.execute(f"ATTACH '{db_path}' AS {pred_alias} (READ_ONLY)")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        SELECT
            p.trade_date,
            p.stock_code,
            p.pred_prob,
            l.{label_col} AS realized
        FROM {pred_alias}."{pred_table}" p
        JOIN labels.prod_l3_prediction_label_parts_current l
          ON p.trade_date = l.trade_date
         AND p.stock_code = l.stock_code
        WHERE p.stock_code NOT LIKE '%.BJ'
          AND l.{label_col} IS NOT NULL
        """
    )
    daily_topn = con.execute(
        f"""
        WITH ranked AS (
            SELECT
                trade_date,
                stock_code,
                pred_prob,
                realized,
                row_number() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code) AS rn
            FROM {view_name}
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
    daily_rank_ic = con.execute(
        f"""
        WITH ranked AS (
            SELECT
                trade_date,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_prob) AS pred_rank,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY realized) AS ret_rank
            FROM {view_name}
        )
        SELECT
            trade_date,
            corr(pred_rank, ret_rank) AS daily_rank_ic
        FROM ranked
        GROUP BY trade_date
        ORDER BY trade_date
        """
    ).fetchdf()
    daily = daily_topn.merge(daily_rank_ic, on="trade_date", how="left", validate="one_to_one")
    for topn in TOP_NS:
        gross_col = f"gross_top{topn}"
        net_col = f"net_top{topn}"
        daily[net_col] = daily[gross_col] - ROUND_TRIP_COST
    return daily, con.execute(f"SELECT COUNT(*) AS rows, MIN(trade_date) AS min_date, MAX(trade_date) AS max_date FROM {view_name}").fetchdf()


def summarize_periods(daily: pd.DataFrame) -> dict[str, dict]:
    dates = daily["trade_date"].astype(str).tolist()
    slices = period_slices(dates)
    out: dict[str, dict] = {}
    for name, period_dates in slices.items():
        part = daily[daily["trade_date"].isin(period_dates)].copy()
        if part.empty:
            continue
        record = {
            "trade_days": int(len(part)),
            "mean_rank_ic": float(part["daily_rank_ic"].mean()),
        }
        for topn in TOP_NS:
            record[f"gross_top{topn}"] = float(part[f"gross_top{topn}"].mean())
            record[f"net_top{topn}"] = float(part[f"net_top{topn}"].mean())
        out[name] = record
    return out


def build_pairwise_top_overlap(con: duckdb.DuckDBPyConnection, left: str, right: str, topn: int) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH a AS (
            SELECT
                trade_date,
                stock_code
            FROM (
                SELECT
                    trade_date,
                    stock_code,
                    row_number() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code) AS rn
                FROM eval_{left}
            )
            WHERE rn <= {topn}
        ),
        b AS (
            SELECT
                trade_date,
                stock_code
            FROM (
                SELECT
                    trade_date,
                    stock_code,
                    row_number() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code) AS rn
                FROM eval_{right}
            )
            WHERE rn <= {topn}
        ),
        joined AS (
            SELECT
                a.trade_date,
                COUNT(*) AS overlap_count
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


def build_unique_top1_vs_10d(con: duckdb.DuckDBPyConnection, horizon: str) -> dict:
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
                    row_number() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code) AS rn
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
                    row_number() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code) AS rn
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


def decide_alpha(rows: list[dict]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    candidate_found = False
    for row in rows:
        if row["horizon"] == "10d":
            continue
        if row["full_net_top3"] <= 0:
            reasons.append(f"{row['horizon']} 全窗口 Top3 成本后为负或接近零。")
            continue
        if row["recent60_net_top3"] <= 0 or row["recent120_net_top3"] <= 0:
            reasons.append(f"{row['horizon']} recent60/recent120 Top3 成本后不稳定。")
            continue
        if row["top5_overlap_vs_10d"] is not None and row["top5_overlap_vs_10d"] >= 0.6:
            reasons.append(f"{row['horizon']} 与 10D 前排重合度较高，不构成低相关独立来源。")
            continue
        if row["top1_diff_days_vs_10d"] > 0 and row["top1_diff_net_vs_10d"] is not None and row["top1_diff_net_vs_10d"] <= 0:
            reasons.append(f"{row['horizon']} 在与 10D 不同 Top1 的日期里成本后仍无正边际。")
            continue
        candidate_found = True
    if candidate_found:
        return "found_research_candidate", reasons
    return "no_independent_alpha_in_current_formal", reasons


def render_md(
    manifest_rows: list[dict],
    horizon_rows: list[dict],
    pair_rows: list[dict],
    verdict: str,
    reasons: list[str],
) -> str:
    lines = [
        "# 当前 active formal L4 独立 Alpha 只读诊断",
        "",
        "## 当前结论",
        "",
    ]
    if verdict == "no_independent_alpha_in_current_formal":
        lines.append("当前 active formal L4 中，没有发现一个同时满足“与现有 3D/5D/10D 前排低相关”且“扣除 0.66% 往返成本后仍跨年度、recent60、recent120 稳定为正”的独立 Alpha 来源。")
    else:
        lines.append("当前 active formal L4 中存在可继续研究的独立 Alpha 线索，但本报告不做 formal 发布建议。")
    lines.extend(
        [
            "",
            "## 可供策略侧继续引用的正式 manifest",
            "",
        ]
    )
    for row in manifest_rows:
        lines.append(f"- `{row['label']}`: `{row['manifest_path']}`")
    lines.extend(
        [
            "",
            "## 各期限成本后前排摘要",
            "",
            "| horizon | full net top1 | full net top3 | full net top5 | recent120 net top3 | recent60 net top3 | mean rankIC | top5 overlap vs 10D | top1 diff days vs 10D | top1 diff net vs 10D |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in horizon_rows:
        overlap = "" if row["top5_overlap_vs_10d"] is None else f"{row['top5_overlap_vs_10d']:.4f}"
        diff_net = "" if row["top1_diff_net_vs_10d"] is None else f"{row['top1_diff_net_vs_10d']:.4%}"
        lines.append(
            f"| {row['horizon']} | {row['full_net_top1']:.4%} | {row['full_net_top3']:.4%} | {row['full_net_top5']:.4%} | "
            f"{row['recent120_net_top3']:.4%} | {row['recent60_net_top3']:.4%} | {row['full_rank_ic']:.4f} | {overlap} | "
            f"{row['top1_diff_days_vs_10d']} | {diff_net} |"
        )
    lines.extend(
        [
            "",
            "## 相关性与重合度",
            "",
            "| pair | full top1 overlap | recent120 top1 overlap | recent60 top1 overlap | full top3 overlap | recent60 top3 overlap | full top5 overlap | recent60 top5 overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pair_rows:
        lines.append(
            f"| {row['pair']} | {row['full_top1_overlap']:.4f} | {row['recent120_top1_overlap']:.4f} | {row['recent60_top1_overlap']:.4f} | "
            f"{row['full_top3_overlap']:.4f} | {row['recent60_top3_overlap']:.4f} | {row['full_top5_overlap']:.4f} | {row['recent60_top5_overlap']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 诊断要点",
            "",
        ]
    )
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本次只读诊断未训练、未调参、未改 formal manifest、未改 L4 数值表、未生成信号、未跑策略回测。",
            "- `20260722` 之后未见前向窗口未打开；本报告只使用成熟标签窗口做模型侧事实判断。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {h: load_manifest(h) for h in MANIFESTS}
    con = duckdb.connect()
    con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS labels (READ_ONLY)")
    daily_map: dict[str, pd.DataFrame] = {}
    horizon_rows: list[dict] = []
    manifest_rows: list[dict] = []
    try:
        for horizon, meta in manifests.items():
            daily, coverage = build_daily_metrics(con, horizon, meta)
            daily_map[horizon] = daily
            periods = summarize_periods(daily)
            horizon_rows.append(
                {
                    "horizon": horizon,
                    "manifest_path": meta["manifest_path"],
                    "db_path": meta["db_path"],
                    "table": meta["table"],
                    "mature_min_date": str(coverage.loc[0, "min_date"]),
                    "mature_max_date": str(coverage.loc[0, "max_date"]),
                    "mature_rows": int(coverage.loc[0, "rows"]),
                    "full_rank_ic": float(periods["2022H2"]["mean_rank_ic"] * 0 + daily["daily_rank_ic"].mean()),
                    "full_net_top1": float(daily["net_top1"].mean()),
                    "full_net_top3": float(daily["net_top3"].mean()),
                    "full_net_top5": float(daily["net_top5"].mean()),
                    "recent120_net_top3": float(periods["recent120"]["net_top3"]),
                    "recent60_net_top3": float(periods["recent60"]["net_top3"]),
                    "periods": periods,
                    "top1_diff_days_vs_10d": 0,
                    "top1_diff_net_vs_10d": None,
                    "top1_overlap_vs_10d": None,
                    "top3_overlap_vs_10d": None,
                    "top5_overlap_vs_10d": None,
                }
            )
            manifest_rows.append(
                {
                    "label": meta["label"],
                    "manifest_path": meta["manifest_path"],
                    "db_path": meta["db_path"],
                    "table": meta["table"],
                    "source_type": meta["source_type"],
                    "approval_status": meta["approval_status"],
                    "pred_prob_sha256": meta["pred_prob_sha256"],
                }
            )

        pair_rows: list[dict] = []
        overlap_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
        for left, right in combinations(MANIFESTS.keys(), 2):
            record = {"pair": f"{left}-{right}"}
            for topn in TOP_NS:
                overlap_df = build_pairwise_top_overlap(con, left, right, topn)
                overlap_cache[(left, right, topn)] = overlap_df
                record[f"full_top{topn}_overlap"] = float(overlap_df["overlap_ratio"].mean())
                dates = overlap_df["trade_date"].astype(str).tolist()
                slices = period_slices(dates)
                record[f"recent120_top{topn}_overlap"] = float(
                    overlap_df[overlap_df["trade_date"].isin(slices["recent120"])]["overlap_ratio"].mean()
                )
                record[f"recent60_top{topn}_overlap"] = float(
                    overlap_df[overlap_df["trade_date"].isin(slices["recent60"])]["overlap_ratio"].mean()
                )
            pair_rows.append(record)
        for row in horizon_rows:
            unique_vs_10d = build_unique_top1_vs_10d(con, row["horizon"])
            row["top1_diff_days_vs_10d"] = unique_vs_10d["different_days"]
            row["top1_diff_net_vs_10d"] = unique_vs_10d["mean_net"]
            if row["horizon"] == "10d":
                continue
            pair_key = tuple(sorted([row["horizon"], "10d"]))
            top1_df = overlap_cache[(pair_key[0], pair_key[1], 1)]
            top3_df = overlap_cache[(pair_key[0], pair_key[1], 3)]
            top5_df = overlap_cache[(pair_key[0], pair_key[1], 5)]
            row["top1_overlap_vs_10d"] = float(top1_df["overlap_ratio"].mean())
            row["top3_overlap_vs_10d"] = float(top3_df["overlap_ratio"].mean())
            row["top5_overlap_vs_10d"] = float(top5_df["overlap_ratio"].mean())
    finally:
        con.close()

    verdict, reasons = decide_alpha(horizon_rows)
    payload = {
        "status": verdict,
        "analysis_date": "2026-07-21",
        "round_trip_cost": ROUND_TRIP_COST,
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
        "strategy_question": "whether current active formal L4 contains a low-correlation, cost-stable independent alpha source for preregistered validation",
    }
    json_path = REPORT_DIR / "independent_alpha_review.json"
    md_path = REPORT_DIR / "独立Alpha诊断报告.md"
    csv_path = REPORT_DIR / "independent_alpha_horizon_summary.csv"
    pair_csv = REPORT_DIR / "independent_alpha_pairwise_summary.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(horizon_rows).drop(columns=["periods"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(pair_rows).to_csv(pair_csv, index=False, encoding="utf-8-sig")
    md_path.write_text(render_md(manifest_rows, horizon_rows, pair_rows, verdict, reasons), encoding="utf-8")
    print(json.dumps({"status": verdict, "json": str(json_path), "md": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
