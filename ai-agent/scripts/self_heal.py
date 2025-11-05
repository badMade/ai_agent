"""Automated maintenance tasks executed by the self-heal workflow."""

from __future__ import annotations

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENT_FILE = PROJECT_ROOT / "agent.py"


def main() -> int:
    """Apply lightweight repository fixes.

    For now the script ensures that ``agent.py`` ends with a newline so that
    version control diffs remain stable across platforms. Additional automated
    maintenance tasks can be added here as the project grows.
    """

    if not AGENT_FILE.exists():
        print("::warning::agent.py not found; nothing to heal.")
        return 0

    contents = AGENT_FILE.read_text(encoding="utf-8")
    if contents.endswith("\n"):
        print("[self-heal] No changes required.")
        return 0

    AGENT_FILE.write_text(contents + "\n", encoding="utf-8")
    print("[self-heal] Added trailing newline to agent.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
