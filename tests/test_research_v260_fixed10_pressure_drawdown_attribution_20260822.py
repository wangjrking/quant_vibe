from quant.main.research_v260_fixed10_pressure_drawdown_attribution_20260822 import (
    CHECKPOINT_PATH,
)


def test_pressure_attribution_uses_current_pressure_checkpoint():
    assert "position_pressure_exit_checkpoint" in str(CHECKPOINT_PATH)
