"""Install the built wheel in fresh core/provider environments on every CI platform."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit("expected exactly one freshly built wheel")
    with tempfile.TemporaryDirectory(prefix="deskagents-wheel-") as directory:
        for extra in ("", "deepseek", "openai"):
            env = Path(directory) / (extra or "core")
            subprocess.run(
                ["uv", "venv", "--no-project", "--python", sys.executable, str(env)], check=True
            )
            python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            requirement = str(wheels[0]) + (f"[{extra}]" if extra else "")
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python), requirement], check=True
            )
            subprocess.run(
                [str(python), "-I", str(root / "scripts/check_installation.py")],
                cwd=directory,
                check=True,
            )
            code = (
                "import importlib.util; assert importlib.util.find_spec('httpx') is None"
                if not extra
                else (
                    f"from deskagents.llm.{extra} import "
                    + ("DeepSeekAdapter" if extra == "deepseek" else "OpenAIAdapter")
                )
            )
            subprocess.run([str(python), "-I", "-c", code], cwd=directory, check=True)
            subprocess.run(
                [str(python), "-I", str(root / "examples/minimal_agent.py")],
                cwd=directory,
                check=True,
            )
            subprocess.run(
                [str(python), "-I", str(root / "examples/durable_agent.py"), str(env / "sessions")],
                cwd=directory,
                check=True,
            )


if __name__ == "__main__":
    main()
