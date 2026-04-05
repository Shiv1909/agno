"""E2BSandbox: BaseSandbox implementation using E2B Code Interpreter."""

from __future__ import annotations

from typing import Any, Optional

try:
    from e2b_code_interpreter import Sandbox as E2BSdk
except ImportError:
    raise ImportError(
        "`e2b_code_interpreter` not installed. Install with: pip install e2b_code_interpreter"
    )

from agno.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from agno.backends.sandbox import BaseSandbox
from agno.utils.log import logger


class E2BSandbox(BaseSandbox):
    """Sandbox backend backed by E2B Code Interpreter cloud sandboxes.

    Implements execute(), upload_files(), download_files(), and id.
    All file operations (ls, read, write, edit, grep, glob) are inherited
    from BaseSandbox and run via execute() inside the E2B sandbox.

    Args:
        api_key: E2B API key. Defaults to E2B_API_KEY environment variable.
        timeout: Sandbox lifetime in seconds. Default: 300 (5 minutes).
        sandbox_options: Additional kwargs passed to E2B Sandbox constructor.

    Example::

        from agno.backends.e2b import E2BSandbox
        from agno.tools.backend import BackendToolkit
        from agno.agent import Agent

        agent = Agent(tools=[BackendToolkit(E2BSandbox(api_key="..."))])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 300,
        sandbox_options: Optional[dict[str, Any]] = None,
    ) -> None:
        from os import getenv
        self._api_key = api_key or getenv("E2B_API_KEY")
        if not self._api_key:
            raise ValueError("E2B_API_KEY not set. Pass api_key= or set the E2B_API_KEY env var.")

        options = sandbox_options or {}
        self._sandbox = E2BSdk.create(api_key=self._api_key, timeout=timeout, **options)
        logger.debug(f"E2BSandbox: created sandbox {self._sandbox.sandbox_id}")

    @property
    def id(self) -> str:
        return getattr(self._sandbox, "sandbox_id", str(id(self._sandbox)))

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """Run a shell command inside the E2B sandbox."""
        try:
            kwargs: dict[str, Any] = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            result = self._sandbox.commands.run(command, **kwargs)
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append("\n".join(f"[stderr] {line}" for line in result.stderr.split("\n") if line))
            output = "\n".join(output_parts) if output_parts else "<no output>"
            exit_code = result.exit_code if hasattr(result, "exit_code") else 0
            return ExecuteResponse(output=output, exit_code=exit_code)
        except Exception as e:
            return ExecuteResponse(output=f"Error executing command: {e}", exit_code=1)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                import io
                self._sandbox.files.write(path, io.BytesIO(content))
                responses.append(FileUploadResponse(path=path))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self._sandbox.files.read(path)
                if isinstance(content, str):
                    content = content.encode("utf-8")
                responses.append(FileDownloadResponse(path=path, content=content))
            except Exception as e:
                responses.append(FileDownloadResponse(path=path, error=str(e)))
        return responses

    def close(self) -> None:
        """Shut down the E2B sandbox."""
        try:
            self._sandbox.kill()
        except Exception:
            pass
