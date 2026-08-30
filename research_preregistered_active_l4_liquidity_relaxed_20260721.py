from pathlib import Path

import research_preregistered_active_l4_horizon_rotation_20260721 as engine


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_liquidity_relaxed_preregistration_20260721"

engine.REPORT_DIR = REPORT_DIR
engine.PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
engine.STAGE1_PATH = REPORT_DIR / "stage1_observation_screen.csv"
engine.STAGE2_PATH = REPORT_DIR / "stage2_observation_screen.csv"
engine.FROZEN_PATH = REPORT_DIR / "frozen_candidates_before_validation.json"
engine.VALIDATION_PATH = REPORT_DIR / "final_validation_once.csv"
engine.SUMMARY_PATH = REPORT_DIR / "research_summary.json"


if __name__ == "__main__":
    engine.main()
