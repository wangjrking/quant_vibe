import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "full_history_initialization_policy_v1_20260830.json"


def test_full_history_initialization_is_isolated_from_daily_incremental_workflow() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    assert policy["workflow_type"] == "full_history_initialization"
    assert policy["activation"]["requires_explicit_owner_authorization"] is True
    assert policy["activation"]["must_not_reuse_incremental_run_id_or_monitor"] is True
    assert policy["boundaries"]["incremental_workflow_must_remain_target_date_only"] is True


def test_full_history_initialization_stops_before_strategy_and_trading() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    stages = policy["stages"]

    assert [stage["id"] for stage in stages] == ["F0", "F1", "F2", "F3", "F4"]
    assert policy["boundaries"]["l5_l6_l7_l8_automatic_progression"] is False
    assert policy["boundaries"]["live_trade_or_external_publication"] is False
    assert policy["audit"]["required_before_each_active_cutover"] is True
