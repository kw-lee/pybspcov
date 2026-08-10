import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = PROJECT_ROOT / "examples" / "quickstart.py"


def test_quickstart_runs_bm_and_sbm_end_to_end() -> None:
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, str(QUICKSTART)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "BMSPCov covariance shape: (3, 3)" in result.stdout
    assert "SBMSPCov covariance shape: (3, 3)" in result.stdout
    assert "SBMSPCov active edges:" in result.stdout
