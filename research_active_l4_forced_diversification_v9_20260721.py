from pathlib import Path

import research_active_l4_diversified_v7_20260721 as engine


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_forced_diversification_v9_20260721"

engine.REPORT_DIR = REPORT_DIR
engine.PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
engine.STAGE1_PATH = REPORT_DIR / "stage1.csv"
engine.STAGE2_PATH = REPORT_DIR / "stage2.csv"
engine.FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
engine.SUMMARY_PATH = REPORT_DIR / "research_summary.json"
engine.ACTION_DIR = REPORT_DIR / "juejin_actions"
engine.ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"


if __name__ == "__main__":
    engine.main()
