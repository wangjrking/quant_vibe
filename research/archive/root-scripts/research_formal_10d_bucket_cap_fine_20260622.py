from __future__ import annotations

import csv
import json
from pathlib import Path

import research_formal_10d_bucket_cap_grid_20260622 as base


REPORT_DIR = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_bucket_cap_fine_20260622"
)


def _variant(
    name: str,
    *,
    strong: float,
    weak: float,
    low_amount: float,
    weak_rank35: float,
    gap0609: float,
    mv50_100: float,
    cap: float,
    holding_days: int = 6,
    max_positions: int = 6,
    eqdd: str = "strict",
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = dict(base.BASE_ENV)
    env.update(base.EQDD_PROFILES[eqdd])
    if extra_env:
        env.update(extra_env)
    return {
        "name": name,
        "strong": strong,
        "weak": weak,
        "low_amount": low_amount,
        "weak_rank35": weak_rank35,
        "gap0609": gap0609,
        "mv50_100": mv50_100,
        "cap": cap,
        "holding_days": holding_days,
        "max_positions": max_positions,
        "target_position_pct": cap,
        "eqdd": eqdd,
        "env": env,
    }


VARIANTS = [
    _variant(
        f"strict_s{str(strong).replace('.', 'p')}_w{str(weak).replace('.', 'p')}_la{str(low_amount).replace('.', 'p')}_cap42",
        strong=strong,
        weak=weak,
        low_amount=low_amount,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
    )
    for strong in (0.32, 0.325, 0.33)
    for weak in (0.20, 0.205, 0.21)
    for low_amount in (0.42, 0.43)
]

VARIANTS += [
    _variant(
        "strict_s32_w20_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.20,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s325_w20_la42_bad14_cap42_pos7",
        strong=0.325,
        weak=0.20,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s32_w205_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.205,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s325_w205_la42_bad14_cap42_pos7",
        strong=0.325,
        weak=0.205,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s325_w206_la42_bad14_cap42_pos7",
        strong=0.325,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s325_w207_la42_bad14_cap42_pos7",
        strong=0.325,
        weak=0.207,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s32_w206_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "base_s32_w206_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s32_w2065_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.2065,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s32_w2061_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.2061,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s32_w2062_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.2062,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s32_w2063_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.2063,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s32_w20625_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.20625,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s32_w20627_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.20627,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s32_w20629_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.20629,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s32_w2064_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.2064,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s32_w207_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.207,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s321_w206_la42_bad14_cap42_pos7",
        strong=0.321,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "base_s32_w206_la42_bad14_cap43_pos7",
        strong=0.32,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.43,
        max_positions=7,
        eqdd="base",
    ),
    _variant(
        "mid_s32_w206_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="mid",
    ),
    _variant(
        "mid_s32_w206_la42_bad14_cap42_pos7_no_score_exit",
        strong=0.32,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="mid",
        extra_env={"GM_OPEN_DAILY_SCORE_EXIT": "0"},
    ),
    _variant(
        "mid_s32_w206_la42_bad14_cap42_pos7_no_dd",
        strong=0.32,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="mid",
        extra_env={"GM_EQUITY_DD_RISK_MODE": "0"},
    ),
    _variant(
        "base_s32_w2063_la42_bad14_cap42_pos7_no_score_exit",
        strong=0.32,
        weak=0.2063,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        eqdd="base",
        extra_env={"GM_OPEN_DAILY_SCORE_EXIT": "0"},
    ),
    _variant(
        "ultra_s32_w206_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        extra_env={
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.07",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
            "GM_EQUITY_DD_SOFT_SCALE": "0.76",
            "GM_EQUITY_DD_HARD_SCALE": "0.50",
        },
    ),
    _variant(
        "strict_s32_w206_la42_bad14_cap42_pos7_score99",
        strong=0.32,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.99", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3"},
    ),
    _variant(
        "strict_s32_w206_la42_bad14_cap42_pos7_score98",
        strong=0.32,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
        extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.98", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3"},
    ),
    _variant(
        "strict_s318_w206_la42_bad14_cap42_pos7",
        strong=0.318,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s319_w206_la42_bad14_cap42_pos7",
        strong=0.319,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s321_w206_la42_bad14_cap42_pos7",
        strong=0.321,
        weak=0.206,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s32_w2055_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.2055,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s32_w2065_la42_bad14_cap42_pos7",
        strong=0.32,
        weak=0.2065,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s319_w2065_la42_bad14_cap42_pos7",
        strong=0.319,
        weak=0.2065,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s321_w2065_la42_bad14_cap42_pos7",
        strong=0.321,
        weak=0.2065,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.42,
        max_positions=7,
    ),
    _variant(
        "strict_s32_w20_la42_bad14_cap43_pos7",
        strong=0.32,
        weak=0.20,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.43,
        max_positions=7,
    ),
    _variant(
        "strict_s325_w20_la42_bad14_cap43_pos7",
        strong=0.325,
        weak=0.20,
        low_amount=0.42,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.43,
        max_positions=7,
    ),
    _variant(
        "strict_s33_w21_la44_bad15_cap44",
        strong=0.33,
        weak=0.21,
        low_amount=0.44,
        weak_rank35=0.15,
        gap0609=0.21,
        mv50_100=0.21,
        cap=0.44,
    ),
    _variant(
        "mid_s33_w21_la44_bad15_cap44",
        strong=0.33,
        weak=0.21,
        low_amount=0.44,
        weak_rank35=0.15,
        gap0609=0.21,
        mv50_100=0.21,
        cap=0.44,
        eqdd="mid",
    ),
    _variant(
        "strict_s33_w21_la44_bad15_cap44_h7",
        strong=0.33,
        weak=0.21,
        low_amount=0.44,
        weak_rank35=0.15,
        gap0609=0.21,
        mv50_100=0.21,
        cap=0.44,
        holding_days=7,
    ),
    _variant(
        "strict_s32_w21_la44_bad14_cap44_pos7",
        strong=0.32,
        weak=0.21,
        low_amount=0.44,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.44,
        max_positions=7,
    ),
    _variant(
        "strict_s32_w205_la43_bad14_cap43_score99",
        strong=0.32,
        weak=0.205,
        low_amount=0.43,
        weak_rank35=0.14,
        gap0609=0.20,
        mv50_100=0.20,
        cap=0.43,
        extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.99", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3"},
    ),
]


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric(row: dict, key: str) -> float:
    return base._to_float(row.get(key), float("-inf"))


def _objective(row: dict) -> float:
    return min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80)


def main() -> int:
    base.REPORT_DIR = REPORT_DIR
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        base._write_signal(cfg, signal_file)
        returncode = base._run_backtest(cfg, signal_file, log_file)
        indicator = base._extract_indicator(log_file) or {}
        row = {
            "name": cfg["name"],
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "config_json": json.dumps({key: value for key, value in cfg.items() if key != "env"}, ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(base._exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {cfg['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}"
        )
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_avg80_by_sharpe.csv",
        [
            row
            for row in sorted(results, key=lambda item: _metric(item, "sharpe"), reverse=True)
            if _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
