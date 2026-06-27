from __future__ import annotations

import csv
import json
from pathlib import Path

import research_formal_5d10d_attribution_scale_refine_20260622 as base


REPORT_DIR = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_effective_end_position_refine_20260622"
)
BASE_SIGNAL = (
    base.MAIN
    / "strategy_library"
    / "production"
    / "prod_formal_5d10d_gap_v20260622"
    / "signals"
    / "backtest_signals_gap_t0p03_g1p08_b0p88.csv"
)
BACKTEST_END = "2026-06-18 15:30:00"


def _rules(good_threshold: float, good_scale: float, bad_scale: float) -> list[dict]:
    return [
        {"field": "pred_gap", "op": "lt", "threshold": good_threshold, "scale": good_scale},
        {"field": "pred_gap", "op": "between", "lower": 0.05, "upper": 0.10, "scale": bad_scale},
    ]


def _variant(
    name: str,
    good_threshold: float,
    good_scale: float,
    bad_scale: float,
    max_single: float,
    max_positions: int = 5,
    holding_days: int = 7,
    max_holding_days: int = 10,
    min_day_target: float | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict:
    return {
        "name": name,
        "rules": _rules(good_threshold, good_scale, bad_scale),
        "normalize_day": True,
        "min_day_target": min_day_target,
        "max_single_position_pct": max_single,
        "target_position_pct": max_single,
        "max_positions": max_positions,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "extra_env": extra_env or {},
    }


VARIANTS = [
    _variant("eff_current_cap28", 0.030, 1.08, 0.88, 0.28),
    _variant("eff_best_sharpe_cap28", 0.030, 1.08, 0.82, 0.28),
    _variant("eff_highannual_cap28", 0.015, 1.08, 0.94, 0.28),
]

for threshold in [0.015, 0.020, 0.025, 0.030]:
    for good_scale in [1.08, 1.12]:
        for bad_scale in [0.82, 0.88]:
            for cap in [0.30, 0.32, 0.35]:
                VARIANTS.append(
                    _variant(
                        f"eff_gap_t{str(threshold).replace('.', 'p')}_g{str(good_scale).replace('.', 'p')}_b{str(bad_scale).replace('.', 'p')}_cap{str(cap).replace('.', 'p')}",
                        threshold,
                        good_scale,
                        bad_scale,
                        cap,
                    )
                )

for cap in [0.30, 0.32, 0.35]:
    VARIANTS.extend(
        [
            _variant(f"eff_current_cap{str(cap).replace('.', 'p')}_noeqdd", 0.030, 1.08, 0.88, cap, extra_env={"GM_EQUITY_DD_RISK_MODE": "0"}),
            _variant(
                f"eff_current_cap{str(cap).replace('.', 'p')}_eqdd_loose",
                0.030,
                1.08,
                0.88,
                cap,
                extra_env={
                    "GM_EQUITY_DD_SOFT_TRIGGER": "0.14",
                    "GM_EQUITY_DD_HARD_TRIGGER": "0.24",
                    "GM_EQUITY_DD_SOFT_SCALE": "0.92",
                    "GM_EQUITY_DD_HARD_SCALE": "0.75",
                },
            ),
            _variant(
                f"eff_current_cap{str(cap).replace('.', 'p')}_defer10_pos",
                0.030,
                1.08,
                0.88,
                cap,
                extra_env={
                    "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1",
                    "GM_NO_SIGNAL_MAX_HOLDING_DAYS": "10",
                    "GM_DEFER_EXIT_MIN_POSITION_RETURN": "0.02",
                },
            ),
        ]
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric(row: dict, key: str) -> float:
    return base._to_float(row.get(key), float("-inf"))


def _write_report(rows: list[dict]) -> None:
    hits = [
        row
        for row in rows
        if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
    ]
    best_objective = max(
        rows,
        key=lambda row: min(
            _metric(row, "annual") / 3.0,
            _metric(row, "sharpe") / 4.0,
            _metric(row, "avg_invested_pct") / 0.80,
        ),
    )
    best_annual = max(rows, key=lambda row: _metric(row, "annual"))
    best_sharpe = max(rows, key=lambda row: _metric(row, "sharpe"))
    report = f"""# 5D10D 有效截止日仓位调参结论

## 当前结论

本轮只使用标准链路 formal L4 的 5D+10D 预测资产，回测截止日对齐到当前可验证行情与预测覆盖上限 `2026-06-18`。未使用 1D、行业限制、月份排除或日期排除。

- 候选数量：{len(rows)}
- 达到 `年化 >= 3.0`、`夏普 >= 4.0`、`平均仓位 >= 0.80` 的候选数量：{len(hits)}
- 是否建议替换当前 L5：{"是" if hits else "否"}

## 最优候选

- 综合最接近目标：`{best_objective["name"]}`，年化 `{_metric(best_objective, "annual"):.6f}`，夏普 `{_metric(best_objective, "sharpe"):.6f}`，平均仓位 `{_metric(best_objective, "avg_invested_pct"):.6f}`。
- 最高年化：`{best_annual["name"]}`，年化 `{_metric(best_annual, "annual"):.6f}`，夏普 `{_metric(best_annual, "sharpe"):.6f}`，平均仓位 `{_metric(best_annual, "avg_invested_pct"):.6f}`。
- 最高夏普：`{best_sharpe["name"]}`，年化 `{_metric(best_sharpe, "annual"):.6f}`，夏普 `{_metric(best_sharpe, "sharpe"):.6f}`，平均仓位 `{_metric(best_sharpe, "avg_invested_pct"):.6f}`。

## 证据路径

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- 信号目录：`{REPORT_DIR / "signals"}`
- GM 日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "有效截止日仓位调参结论.md").write_text(report, encoding="utf-8")


def main() -> int:
    base.REPORT_DIR = REPORT_DIR
    base.BACKTEST_END = BACKTEST_END
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    base_rows = base._load_rows(BASE_SIGNAL)
    keys = {(str(row.get("signal_date")), str(row.get("stock_code"))) for row in base_rows}
    fusion_features = base._load_fusion_features(keys)
    results: list[dict] = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        base._write_variant_signal(variant, signal_file, base_rows, fusion_features)
        returncode = base._run_backtest(variant, signal_file, log_file)
        indicator = base._extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "rules_json": json.dumps(variant["rules"], ensure_ascii=False, sort_keys=True),
            "max_single_position_pct": variant["max_single_position_pct"],
            "target_position_pct": variant["target_position_pct"],
            "max_positions": variant["max_positions"],
            "holding_days": variant["holding_days"],
            "max_holding_days": variant["max_holding_days"],
            "extra_env_json": json.dumps(variant["extra_env"], ensure_ascii=False, sort_keys=True),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(base._signal_stats(signal_file))
        row.update(base._exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} "
            f"avg_inv={row.get('avg_invested_pct')}"
        )
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual"), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    _write_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
