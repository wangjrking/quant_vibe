"""Read-only 2022-2024 front-rank diagnosis for strict PIT 3D/5D baselines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_expanding_pit_oof_baselines_20260829 import canonical_frame_hash, sha256_file


ROOT = Path("quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3")
OUT = Path("quant/data_file/reports/model_agent_3d_5d_front_rank_diagnostics_20260830")
HORIZONS = ("3d", "5d")


def daily_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        ranked = group.sort_values(["pred_prob", "stock_code"], ascending=[False, True], kind="mergesort")
        all_mean = float(ranked["target"].mean())
        scores = ranked["pred_prob"].to_numpy(dtype="float64")
        row: dict[str, object] = {
            "trade_date": str(trade_date),
            "rank_ic": float(ranked["pred_prob"].rank(method="average").corr(ranked["target"].rank(method="average"))),
            "distinct_score_count": int(ranked["pred_prob"].nunique()),
            "score_std": float(np.std(scores)),
        }
        for count in (1, 3, 5, 10, 20):
            row[f"top{count}_excess"] = float(ranked.head(count)["target"].mean() - all_mean)
        rows.append(row)
    return pd.DataFrame(rows)


def mean_metric(frame: pd.DataFrame) -> dict[str, float | int]:
    result: dict[str, float | int] = {"trade_days": int(len(frame))}
    for column in ("rank_ic", "distinct_score_count", "score_std", "top1_excess", "top3_excess", "top5_excess", "top10_excess", "top20_excess"):
        result[column] = float(frame[column].mean())
    return result


def run() -> int:
    if OUT.exists():
        raise RuntimeError("blocked_existing_output")
    OUT.mkdir(parents=True)
    result: dict[str, object] = {
        "asset_role": "research_only_l4_readonly_diagnostic",
        "development_window": ["20220101", "20241231"],
        "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
        "production_unchanged": True,
        "horizons": {},
    }
    for horizon in HORIZONS:
        path = ROOT / f"{horizon}_oof.parquet"
        source = pd.read_parquet(path)
        frame = source.loc[
            source["fold_id"].isin(["fold2022", "fold2023", "fold2024"])
            & source["label_mature_within_dev"]
            & source["trade_date"].astype(str).str.startswith(("2022", "2023", "2024"))
        ].copy()
        if frame.empty or frame.duplicated(["trade_date", "stock_code"]).any():
            raise RuntimeError(f"blocked_{horizon}_key_domain")
        if frame["stock_code"].astype(str).str.endswith(".BJ").any() or not np.isfinite(frame[["target", "pred_prob"]].to_numpy(dtype="float64")).all():
            raise RuntimeError(f"blocked_{horizon}_quality")
        daily = daily_metrics(frame)
        daily["year"] = daily["trade_date"].str[:4]
        monthly_frame = daily.assign(month=daily["trade_date"].str[:6])
        monthly = {str(month): mean_metric(group) for month, group in monthly_frame.groupby("month", sort=True)}
        by_year = {str(year): mean_metric(group) for year, group in daily.groupby("year", sort=True)}
        result["horizons"][horizon] = {
            "source_oof": str(path),
            "source_oof_sha256": sha256_file(path),
            "same_key_rows": int(len(frame)),
            "duplicate_key_groups": int(frame.duplicated(["trade_date", "stock_code"]).sum()),
            "bj_rows": int(frame["stock_code"].astype(str).str.endswith(".BJ").sum()),
            "null_or_nonfinite_rows": int((~np.isfinite(frame[["target", "pred_prob"]].to_numpy(dtype="float64")).all(axis=1)).sum()),
            "score_sha256": canonical_frame_hash(frame, ["trade_date", "stock_code", "pred_prob"]),
            "aggregate": mean_metric(daily),
            "by_year": by_year,
            "by_month": monthly,
            "front_rank_interpretation": "A candidate must not claim improvement from RankIC alone: each fold must also preserve or improve Top1/Top3/Top5/Top10 under the same daily key domain.",
        }
        daily.to_parquet(OUT / f"{horizon}_daily_front_metrics.parquet", index=False)
    (OUT / "diagnostic_summary.json").write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "hash_inventory.json").write_text(json.dumps({
        "script_sha256": sha256_file(Path(__file__)),
        "summary_sha256": sha256_file(OUT / "diagnostic_summary.json"),
        "3d_daily_sha256": sha256_file(OUT / "3d_daily_front_metrics.parquet"),
        "5d_daily_sha256": sha256_file(OUT / "5d_daily_front_metrics.parquet"),
    }, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
