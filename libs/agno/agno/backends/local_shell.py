"""LocalShellBackend: FilesystemBackend + unrestricted local shell execution.

.. warning::
    This backend grants agents full shell access to the host system.
    Only use in trusted local development environments. Never expose to
    untrusted users or in production/server contexts.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from agno.backends.filesystem import FilesystemBackend
from agno.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from agno.backends.utils import MAX_OUTPUT_BYTES, TRUNCATION_NOTICE, combine_output

DEFAULT_TIMEOUT = 120
"""Default shell command timeout in seconds (2 minutes)."""


class LocalShellBackend(FilesystemBackend, SandboxBackendProtocol):
    """Filesystem backend with unrestricted local shell execution.

    Extends FilesystemBackend with execute() via subprocess.run(shell=True).
    All filesystem operations are inherited from FilesystemBackend.

    Args:
        root: Working directory for both filesystem ops and shell commands.
            Defaults to current working directory.
        virtual_mode: See FilesystemBackend. Note: virtual_mode does NOT
            restrict shell execution — execute() can still access any path.
        timeout: Default timeout in seconds for execute(). Default: 120.
        max_output_bytes: Truncate output at this size. Default: 1 MB.
        env: Environment variables for shell commands. If None and
            inherit_env=False, commands start with only PATH.
        inherit_env: If True, inherit all os.environ vars (plus any env
            overrides). Default: False to reduce secret leakage risk.

    .. warning::
        shell=True is intentional — this backend is designed for LLM-controlled
        local coding assistants where the user trusts the execution environment.
        Always use Human-in-the-Loop confirmation for sensitive operations.
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        *,
        virtual_mode: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        env: Optional[dict[str, str]] = None,
        inherit_env: bool = False,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        super().__init__(root=root, virtual_mode=virtual_mode)
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._sandbox_id = f"local-{uuid.uuid4().hex[:8]}"

        if inherit_env:
            self._env = os.environ.copy()
            if env:
                self._env.update(env)
        else:
            # Minimal environment: just PATH
            default_path = os.environ.get("PATH", "/usr/bin:/bin")
            self._env = {"PATH": default_path}
            if env:
                self._env.update(env)

    @property
    def id(self) -> str:
        return self._sandbox_id

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """Execute a shell command directly on the host system.

        Commands run with shell=True in the backend's root directory.
        stdout and stderr are combined; stderr lines are prefixed with [stderr].

        Args:
            command: Shell command to execute.
            timeout: Override the default timeout for this command.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(output="Error: Command must be a non-empty string.", exit_code=1)

        effective_timeout = timeout if timeout is not None else self._timeout
        if effective_timeout <= 0:
            raise ValueError(f"timeout must be positive, got {effective_timeout}")

        try:
            result = subprocess.run(  # noqa: S602
                command,
                shell=True,  # Intentional: designed for LLM-controlled local execution
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=self._env,
                cwd=str(self.root),
                check=False,
            )

            output = combine_output(result.stdout, result.stderr)

            truncated = False
            if len(output.encode("utf-8")) > self._max_output_bytes:
                output = output.encode("utf-8")[: self._max_output_bytes].decode("utf-8", errors="ignore")
                output += TRUNCATION_NOTICE
                truncated = True

            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Error: Command timed out after {effective_timeout}s. Use timeout= to extend.",
                exit_code=124,
            )
        except Exception as e:  # noqa: BLE001
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
            )
