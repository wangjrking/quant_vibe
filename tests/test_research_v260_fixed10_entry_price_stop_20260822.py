import ast
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[1] / "research_v260_runtime/fixed10_risk_event_v110.py"


def test_entry_price_state_is_initialized_and_cleared():
    source = RUNTIME.read_text(encoding="utf-8")
    assert "entry_raw_price[idx] = float(opens[idx])" in source
    assert "entry_raw_price.pop(idx, None)" in source


def test_entry_loss_requires_score_replacement_advantage():
    source = RUNTIME.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
    assert any(node.id == "entry_loss_exit_condition" for node in names)
    assert "and best_unheld - current >= replacement_advantage" in source


def test_entry_stop_is_opt_in():
    source = RUNTIME.read_text(encoding="utf-8")
    assert "entry_price_loss_exit_override=None" in source
    assert "entry_price_loss_exit_override is not None" in source
