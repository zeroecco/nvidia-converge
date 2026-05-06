from __future__ import annotations

import shutil
import subprocess

from .models import CommandResult


class CommandRunner:
    def __init__(self, apply: bool = False, timeout: int = 120):
        self.apply = apply
        self.timeout = timeout
        self.results: list[CommandResult] = []

    def exists(self, name: str) -> bool:
        return shutil.which(name) is not None

    def run(self, command: list[str], *, mutate: bool = False, allow_fail: bool = True, input_text: str | None = None) -> CommandResult:
        if mutate and not self.apply:
            result = CommandResult(command=command, returncode=None, skipped=True, reason="dry-run")
            self.results.append(result)
            return result
        try:
            proc = subprocess.run(
                command,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            result = CommandResult(
                command=command,
                returncode=proc.returncode,
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
            )
        except FileNotFoundError as exc:
            result = CommandResult(command=command, returncode=127, stderr=str(exc))
        except subprocess.TimeoutExpired as exc:
            result = CommandResult(
                command=command,
                returncode=124,
                stdout=(exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "").strip() if isinstance(exc.stderr, str) else "command timed out",
            )
        self.results.append(result)
        if not allow_fail and result.returncode not in (0, None):
            raise RuntimeError(f"command failed: {' '.join(command)}: {result.stderr}")
        return result
