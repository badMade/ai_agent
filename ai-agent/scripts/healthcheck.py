"""Basic repository health checks used by the self-heal workflow."""

from __future__ import annotations

import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENT_FILE = PROJECT_ROOT / "agent.py"


def _log(message: str) -> None:
    """Emit a log line that is easy to pick out in workflow logs."""

    print(f"[healthcheck] {message}")


def main() -> int:
    """Run lightweight validation to ensure the repository is runnable."""

    _log("Starting healthcheck")

    if not AGENT_FILE.exists():
        print("::error::Expected agent.py to exist but it was not found.")
        return 1

    _log("Validating agent.py syntax")
    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", str(AGENT_FILE)],
        check=False,
        capture_output=True,
        text=True,
    )
    if compile_result.stdout:
        print(compile_result.stdout, end="")
    if compile_result.stderr:
        print(compile_result.stderr, end="", file=sys.stderr)
    if compile_result.returncode != 0:
        print("::error::Syntax validation failed for agent.py.")
        return compile_result.returncode

    _log("Healthcheck completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
