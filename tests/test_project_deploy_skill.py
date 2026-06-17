from pathlib import Path
import importlib.util
import json
import os
import tempfile
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "quant-vibe-deploy"
SKILL_MD = SKILL_DIR / "SKILL.md"
BOOTSTRAP = SKILL_DIR / "scripts" / "bootstrap_quant_vibe.py"


def test_deploy_skill_documents_full_reproduction_workflow():
    text = SKILL_MD.read_text(encoding="utf-8")

    assert "name: quant-vibe-deploy" in text
    assert "description:" in text
    assert "bootstrap_quant_vibe.py" in text
    assert "TUSHARE_TOKEN" in text
    assert "production_factor_parts" in text
    assert "prediction_label_parts" in text
    assert "model_predictions" in text
    assert "run_all_a_raw_update.py" in text
    assert "run_pdb_update.py" in text
    assert "run_production_tasks.py" in text


def test_deploy_bootstrap_dry_run_lists_install_data_model_and_strategy_steps():
    result = subprocess.run(
        [sys.executable, str(BOOTSTRAP), "--dry-run", "--skip-pip"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "create virtual environment" in result.stdout
    assert "install requirements" in result.stdout
    assert "copy config.example.json" in result.stdout
    assert "initialize agent session" in result.stdout
    assert "download/update raw data" in result.stdout
    assert "build production factors" in result.stdout
    assert "build prediction labels" in result.stdout
    assert "train/update model predictions" in result.stdout
    assert "generate strategy signals" in result.stdout


def test_deploy_bootstrap_dry_run_uses_only_tracked_python_entrypoints():
    result = subprocess.run(
        [sys.executable, str(BOOTSTRAP), "--dry-run", "--skip-pip"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    tracked_files = set(
        subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.splitlines()
    )
    planned_scripts = {
        token
        for line in result.stdout.splitlines()
        for token in line.split()
        if token.endswith(".py")
    }

    assert planned_scripts <= tracked_files


def test_deploy_bootstrap_writes_agent_session_files():
    spec = importlib.util.spec_from_file_location("bootstrap_quant_vibe", BOOTSTRAP)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    temp_root = REPO_ROOT / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
        data_dir = Path(temp_dir)
        session_path = module.write_agent_session(
            data_dir=data_dir,
            python_path=Path(sys.executable),
            dry_run=False,
        )

        current_path = data_dir / "runtime" / "agent_sessions" / "current_session.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        current = json.loads(current_path.read_text(encoding="utf-8"))

        assert session_path.exists()
        assert current_path.exists()
        assert session["skill"] == "quant-vibe-deploy"
        assert session["deployment_phase"] == "initialized"
        assert session["data_dir"] == str(data_dir)
        assert session["python"] == str(Path(sys.executable))
        assert {agent["id"] for agent in session["agents"]} == {
            "commander",
            "data-agent",
            "model-agent",
            "strategy-agent",
            "audit-agent",
        }
        assert current["session_id"] == session["session_id"]
        assert current["session_file"] == str(session_path)


def test_deploy_bootstrap_check_reports_missing_private_token_without_secret_leak():
    env = os.environ.copy()
    env.pop("TUSHARE_TOKEN", None)

    result = subprocess.run(
        [
            sys.executable,
            str(BOOTSTRAP),
            "--check",
            "--skip-pip",
            "--require-token",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 2
    assert "TUSHARE_TOKEN is required" in combined
    assert "your_token_here" not in combined
