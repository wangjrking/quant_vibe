from __future__ import annotations

import pytest

from research_v260_fixed10_full_investment_tradeoff_20260822 import (
    contribution_share,
)


def test_contribution_share_preserves_signed_attribution() -> None:
    assert contribution_share(0.08, 0.10) == pytest.approx(0.8)
    assert contribution_share(-0.02, 0.10) == pytest.approx(-0.2)


def test_contribution_share_rejects_zero_total() -> None:
    with pytest.raises(ValueError, match="cannot be zero"):
        contribution_share(0.0, 0.0)
