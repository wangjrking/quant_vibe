"""Generate a sealed pre-2025 front-rank diagnostic for 3D and 5D OOF."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3")
OUT = Path("quant/data_file/reports/model_agent_3d_5d_front_rank_diagnostics_20260830_r2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    ordered = frame.sort_values(["trade_date", "stock_code"], kind="mergesort")[columns]
    for row in ordered.itertuples(index=False, name=None):
        digest.update(("|".join(format(value, ".17g") if isinstance(value, float) else str(value) for value in row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def summarize(daily: pd.DataFrame) -> dict[str, float | int]:
    result: dict[str, float | int] = {"trade_days": int(len(daily))}
    for column in ("rank_ic", "distinct_score_count", "score_std", "top1_excess", "top3_excess", "top5_excess", "top10_excess", "top20_excess"):
        result[column] = float(daily[column].mean())
    return result


def daily_metrics(source: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for date, group in source.groupby("trade_date", sort=True):
        ranked = group.sort_values(["pred_prob", "stock_code"], ascending=[False, True], kind="mergesort")
        all_mean = float(ranked["target"].mean())
        record: dict[str, object] = {
            "trade_date": str(date),
            "rank_ic": float(ranked["pred_prob"].rank(method="average").corr(ranked["target"].rank(method="average"))),
            "distinct_score_count": int(ranked["pred_prob"].nunique()),
            "score_std": float(np.std(ranked["pred_prob"].to_numpy(dtype="float64"))),
        }
        for count in (1, 3, 5, 10, 20):
            record[f"top{count}_excess"] = float(ranked.head(count)["target"].mean() - all_mean)
        records.append(record)
    return pd.DataFrame(records)


def run() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("blocked_existing_output")
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "asset_role": "research_only_l4_readonly_diagnostic",
        "selection_window": ["20220101", "20241231"],
        "sealed_windows": {"2025": "not_read", "2026_plus": "not_read"},
        "production_unchanged": True,
        "horizons": {},
    }
    for horizon in ("3d", "5d"):
        path = ROOT / f"{horizon}_oof.parquet"
        raw = pd.read_parquet(path)
        source = raw.loc[raw["fold_id"].isin(["fold2022", "fold2023", "fold2024"]) & raw["label_mature_within_dev"]].copy()
        if source.duplicated(["trade_date", "stock_code"]).any() or source["stock_code"].astype(str).str.endswith(".BJ").any():
            raise RuntimeError(f"blocked_{horizon}_key_quality")
        if not np.isfinite(source[["target", "pred_prob"]].to_numpy(dtype="float64")).all():
            raise RuntimeError(f"blocked_{horizon}_finite")
        daily = daily_metrics(source)
        daily["year"] = daily["trade_date"].str[:4]
        daily["month"] = daily["trade_date"].str[:6]
        daily.to_parquet(OUT / f"{horizon}_daily_front_metrics.parquet", index=False)
        result["horizons"][horizon] = {
            "source_sha256": sha256_file(path),
            "same_key_rows": int(len(source)),
            "duplicate_key_groups": 0,
            "bj_rows": 0,
            "null_or_nonfinite_rows": 0,
            "score_sha256": canonical_frame_hash(source, ["trade_date", "stock_code", "pred_prob"]),
            "aggregate": summarize(daily),
            "by_year": {str(year): summarize(group) for year, group in daily.groupby("year", sort=True)},
            "by_month": {str(month): summarize(group) for month, group in daily.groupby("month", sort=True)},
            "candidate_rule": "A future candidate must improve or preserve RankIC and Top1/Top3/Top5/Top10 per frozen fold; RankIC-only gains are insufficient.",
        }
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
