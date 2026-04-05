"""RunloopSandbox: BaseSandbox implementation using Runloop cloud devboxes."""

from __future__ import annotations

from typing import Any, Optional

try:
    from runloop_api_client import Runloop
except ImportError:
    raise ImportError(
        "`runloop_api_client` not installed. Install with: pip install runloop-api-client"
    )

from agno.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from agno.backends.sandbox import BaseSandbox
from agno.utils.log import logger


class RunloopSandbox(BaseSandbox):
    """Sandbox backend backed by Runloop cloud devboxes.

    Wraps a Runloop devbox. All file operations are inherited from BaseSandbox
    and run via execute() inside the Runloop devbox.

    Args:
        api_key: Runloop API key. Defaults to RUNLOOP_API_KEY env var.
        devbox_id: Reuse an existing devbox by ID. If None, creates a new one.
        blueprint_id: Blueprint to use when creating a new devbox.
        timeout: Command execution timeout in seconds. Default: 1800.

    Example::

        from agno.backends.runloop import RunloopSandbox
        from agno.tools.backend import BackendToolkit
        from agno.agent import Agent

        agent = Agent(tools=[BackendToolkit(RunloopSandbox(api_key="..."))])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        devbox_id: Optional[str] = None,
        blueprint_id: Optional[str] = None,
        timeout: int = 1800,
    ) -> None:
        from os import getenv
        key = api_key or getenv("RUNLOOP_API_KEY")
        self._client = Runloop(bearer_token=key) if key else Runloop()
        self._timeout = timeout

        if devbox_id:
            self._devbox = self._client.devboxes.retrieve(devbox_id)
        else:
            kwargs: dict[str, Any] = {}
            if blueprint_id:
                kwargs["blueprint_id"] = blueprint_id
            self._devbox = self._client.devboxes.create(**kwargs)

        logger.debug(f"RunloopSandbox: using devbox {self._devbox.id}")

    @property
    def id(self) -> str:
        return str(self._devbox.id)

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """Run a shell command in the Runloop devbox."""
        try:
            effective_timeout = timeout or self._timeout
            result = self._client.devboxes.execute_sync(
                self._devbox.id,
                command=command,
                timeout=effective_timeout,
            )
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append("\n".join(f"[stderr] {l}" for l in stderr.split("\n") if l))
            output = "\n".join(output_parts) if output_parts else "<no output>"
            exit_code = getattr(result, "return_code", 0) or 0
            return ExecuteResponse(output=output, exit_code=exit_code)
        except Exception as e:
            return ExecuteResponse(output=f"Error executing command: {e}", exit_code=1)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                import base64
                content_b64 = base64.b64encode(content).decode("ascii")
                cmd = (
                    f"python3 -c \"import base64,os; "
                    f"os.makedirs(os.path.dirname('{path}') or '.', exist_ok=True); "
                    f"open('{path}','wb').write(base64.b64decode('{content_b64}'))\""
                )
                result = self.execute(cmd)
                if result.exit_code == 0:
                    responses.append(FileUploadResponse(path=path))
                else:
                    responses.append(FileUploadResponse(path=path, error=result.output))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                import base64
                result = self.execute(
                    f"python3 -c \"import base64; print(base64.b64encode(open('{path}','rb').read()).decode())\""
                )
                if result.exit_code == 0:
                    raw = base64.b64decode(result.output.strip())
                    responses.append(FileDownloadResponse(path=path, content=raw))
                else:
                    responses.append(FileDownloadResponse(path=path, error=result.output))
            except Exception as e:
                responses.append(FileDownloadResponse(path=path, error=str(e)))
        return responses

    def close(self) -> None:
        """Shut down the Runloop devbox."""
        try:
            self._client.devboxes.shutdown(self._devbox.id)
        except Exception:
            pass
