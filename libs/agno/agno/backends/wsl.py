"""WSLBackend: LocalShellBackend that routes execute() through Windows Subsystem for Linux.

File operations (ls, read, write, edit, grep, glob) use Python's direct disk I/O on the
Windows filesystem as normal. Only execute() is routed through WSL so the agent runs
shell commands in a real Linux environment while still reading/writing files on Windows
disk (accessible inside WSL at /mnt/<drive>/...).

Usage::

    from agno.backends.wsl import WSLBackend
    from agno.tools.backend import BackendToolkit
    from agno.agent import Agent

    backend = WSLBackend(root="C:/workspace", distro="Ubuntu")
    agent = Agent(tools=[BackendToolkit(backend)])
    # execute("python3 script.py") now runs inside WSL Ubuntu
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path, PureWindowsPath
from typing import Optional

from agno.backends.local_shell import LocalShellBackend
from agno.backends.protocol import ExecuteResponse
from agno.backends.utils import MAX_OUTPUT_BYTES, TRUNCATION_NOTICE, combine_output


def windows_to_wsl_path(windows_path: str) -> str:
    """Convert a Windows path to its WSL mount path.

    Examples:
        C:\\Users\\foo  →  /mnt/c/Users/foo
        D:\\projects    →  /mnt/d/projects
        /already/posix  →  /already/posix  (returned as-is)
    """
    p = str(windows_path).replace("\\", "/")
    # Already a POSIX path (e.g. already inside WSL)
    if p.startswith("/"):
        return p
    # Drive letter: C:/... → /mnt/c/...
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    # Relative path — return as-is, WSL will resolve relative to its cwd
    return p


class WSLBackend(LocalShellBackend):
    """LocalShellBackend that executes commands inside WSL (Windows Subsystem for Linux).

    File operations use Python's direct disk I/O on the Windows filesystem.
    Shell commands are routed through ``wsl -- bash -c "..."`` so the agent
    gets a real Linux environment (apt, bash builtins, Linux tools) while
    files remain on the Windows disk and are accessible from both sides.

    Args:
        root: Working directory for file operations and the default cwd for
            shell commands. Accepts a Windows path (``C:\\workspace``) or a
            POSIX path. Defaults to the current working directory.
        distro: WSL distribution name to use (e.g. ``"Ubuntu"``,
            ``"Ubuntu-22.04"``). If None, uses the default WSL distribution.
        wsl_user: Linux username to run commands as inside WSL. If None,
            uses the default WSL user.
        virtual_mode: Confine file operations to ``root``. Does NOT restrict
            shell commands (WSL has full Linux filesystem access).
        timeout: Default command timeout in seconds. Default: 120.
        max_output_bytes: Truncate output at this byte size. Default: 1 MB.
        env: Extra environment variables forwarded into the WSL session.
        inherit_env: If True, forward all ``os.environ`` vars into WSL.

    Example::

        backend = WSLBackend(root="C:/workspace", distro="Ubuntu")
        toolkit = BackendToolkit(backend)
        agent = Agent(tools=[toolkit])
        # agent can now: write files on Windows disk AND run Linux commands
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        *,
        distro: Optional[str] = None,
        wsl_user: Optional[str] = None,
        virtual_mode: bool = False,
        timeout: int = 120,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        env: Optional[dict[str, str]] = None,
        inherit_env: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            virtual_mode=virtual_mode,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            env=env,
            inherit_env=inherit_env,
        )
        self._distro = distro
        self._wsl_user = wsl_user

        # Pre-compute the WSL-side working directory from the Windows root
        self._wsl_cwd = windows_to_wsl_path(str(self.root))

    @property
    def id(self) -> str:
        distro_tag = self._distro or "wsl-default"
        return f"{distro_tag}-{self._sandbox_id}"

    def _build_wsl_command(self, command: str) -> list[str]:
        """Build the full wsl.exe argv list for a given command.

        Avoids --cd flag (incompatible with some WSL versions/builds).
        Instead, cds inside bash before running the command.

        Custom env vars (self._env) are injected as export statements at the
        top of the bash script rather than via subprocess env= so that WSL's
        own RPC service can still inherit the full Windows environment it needs.
        """
        cmd = ["wsl"]
        if self._distro:
            cmd += ["-d", self._distro]
        if self._wsl_user:
            cmd += ["-u", self._wsl_user]

        # Inject custom env vars inside bash, not via subprocess env=
        env_exports = ""
        if self._env:
            exports = "; ".join(
                f"export {k}={shlex.quote(v)}"
                for k, v in self._env.items()
                if k != "PATH"  # PATH is handled by WSL itself
            )
            if exports:
                env_exports = exports + "; "

        bash_script = f"{env_exports}cd {shlex.quote(self._wsl_cwd)} && {command}"
        cmd += ["bash", "-c", bash_script]
        return cmd

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """Execute a shell command inside WSL.

        The command runs via ``wsl [-d distro] [-u user] bash -c "cd <wsl_cwd> && <command>"``.
        stdout and stderr are combined; stderr lines are prefixed with ``[stderr]``.

        Args:
            command: Shell command to run inside WSL.
            timeout: Override the default timeout for this command.

        Returns:
            ExecuteResponse with output, exit_code, and truncated flag.
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(output="Error: Command must be a non-empty string.", exit_code=1)

        effective_timeout = timeout if timeout is not None else self._timeout
        if effective_timeout <= 0:
            raise ValueError(f"timeout must be positive, got {effective_timeout}")

        full_cmd = self._build_wsl_command(command)

        try:
            result = subprocess.run(
                full_cmd,
                shell=False,   # No Windows shell needed — wsl.exe handles it
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=None,      # Always inherit full Windows env — WSL needs SYSTEMROOT,
                               # WINDIR, etc. for its RPC service connection. Custom vars
                               # are injected via bash exports inside the script instead.
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
                output=f"Error: WSL command timed out after {effective_timeout}s. Use timeout= to extend.",
                exit_code=124,
            )
        except FileNotFoundError:
            return ExecuteResponse(
                output=(
                    "Error: wsl.exe not found. Make sure WSL is installed and "
                    "'wsl' is on your PATH. Run: wsl --install"
                ),
                exit_code=1,
            )
        except Exception as e:  # noqa: BLE001
            return ExecuteResponse(
                output=f"Error executing WSL command ({type(e).__name__}): {e}",
                exit_code=1,
            )
