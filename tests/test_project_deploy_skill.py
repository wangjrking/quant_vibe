from pathlib import Path
import os
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
