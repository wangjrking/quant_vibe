from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_currentcoverage_best_validation_20260720"
    / "validation.json"
)
OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_metric_semantics_audit_20260721"


def indicator(path: Path) -> dict:
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" in line:
            raw = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
            result = {}
            for key in ("pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown"):
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", raw)
                if match:
                    result[key] = float(match.group(1))
            if "pnl_ratio" in result and "pnl_ratio_annual" in result:
                return result
    raise RuntimeError(f"indicator not found: {path}")


def main() -> None:
    source = json.loads(VALIDATION.read_text(encoding="utf-8-sig"))
    run = next(item for item in source["results"] if item["name"] == "full_repeat_a")
    metrics = indicator(Path(run["log_file"]))
    start = datetime.fromisoformat(run["start"])
    end = datetime.fromisoformat(run["end"])
    actual_years = (end - start).total_seconds() / (365.0 * 86400.0)
    platform_inclusive_days = (end.date() - start.date()).days + 1
    platform_years = platform_inclusive_days / 365.0
    cumulative = float(metrics["pnl_ratio"])
    gm_annual = float(metrics["pnl_ratio_annual"])
    platform_linear_annual = cumulative / platform_years
    cagr = (1.0 + cumulative) ** (1.0 / actual_years) - 1.0
    target_platform_annual = 5.0
    target_cumulative = target_platform_annual * platform_years
    target_cagr = (1.0 + target_cumulative) ** (1.0 / platform_years) - 1.0
    result = {
        "status": "metric_semantics_verified_from_juejin_log",
        "source_log": run["log_file"],
        "start": run["start"],
        "end": run["end"],
        "actual_duration_years_365": actual_years,
        "platform_inclusive_calendar_days": platform_inclusive_days,
        "platform_linear_years_365": platform_years,
        "juejin_pnl_ratio": cumulative,
        "juejin_pnl_ratio_annual": gm_annual,
        "juejin_pnl_ratio_annual_label": "平台线性年化/非 CAGR",
        "platform_linear_annual_from_cumulative": platform_linear_annual,
        "compound_annual_growth_rate": cagr,
        "absolute_difference_gm_vs_platform_linear": abs(gm_annual - platform_linear_annual),
        "target_platform_linear_annual": target_platform_annual,
        "target_cumulative_return_for_same_platform_span": target_cumulative,
        "target_cagr_for_same_platform_span": target_cagr,
        "conclusion": "Juejin pnl_ratio_annual is empirically consistent with linear annualization of cumulative return for this run and is not CAGR",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metric_semantics_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 掘金年化指标语义审计

## 结论

本次完整掘金日志中：

- 累计收益率 `pnl_ratio`：`{cumulative:.4%}`
- 掘金年化字段 `pnl_ratio_annual`：`{gm_annual:.4%}`，治理标签为“平台线性年化/非 CAGR”
- 平台起止日期两端计入：`{platform_inclusive_days}` 天，即 `{platform_years:.6f}` 年
- 累计收益按平台计时口径线性折算：`{platform_linear_annual:.4%}`
- 实际时分秒跨度：`{actual_years:.6f}` 年
- 按净值复合计算的 CAGR：`{cagr:.4%}`

掘金年化字段与平台计时口径的线性折算值仅相差 `{abs(gm_annual - platform_linear_annual):.10%}`，但与 CAGR 差异明显。因此，该字段在本次回测中不是复合年化。

同一平台时间跨度下，准入门槛 `pnl_ratio_annual >= 500%` 对应累计收益约 `{target_cumulative:.4%}`，对应 CAGR 约 `{target_cagr:.4%}`。该换算只解释指标，不降低或替代原门槛。

## 后续强制展示口径

所有策略回测报告必须同时展示：

1. 掘金原始 `pnl_ratio_annual`，明确标注“平台线性年化/非 CAGR”；
2. 掘金原始累计收益 `pnl_ratio`；
3. 根据回测起止时间和最终净值计算的 CAGR，并注明时间基准；
4. Sharpe 与最大回撤。

用户指定的 `掘金年化 >= 500%` 仍按掘金原始字段执行准入，但不得把它描述成 CAGR，也不得用线性年化替代复合收益解释。
"""
    (OUT / "metric_semantics_audit.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
