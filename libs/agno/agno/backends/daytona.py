"""DaytonaSandbox: BaseSandbox implementation using Daytona cloud sandboxes."""

from __future__ import annotations

from typing import Any, Optional

try:
    from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig
except ImportError:
    raise ImportError(
        "`daytona` not installed. Install with: pip install daytona"
    )

from agno.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from agno.backends.sandbox import BaseSandbox
from agno.utils.log import logger


class DaytonaSandbox(BaseSandbox):
    """Sandbox backend backed by Daytona cloud sandboxes.

    Implements execute(), upload_files(), download_files(), and id.
    All file operations (ls, read, write, edit, grep, glob) are inherited
    from BaseSandbox and run via execute() inside the Daytona sandbox.

    Args:
        api_key: Daytona API key. Defaults to DAYTONA_API_KEY env var.
        api_url: Daytona API URL. Defaults to DAYTONA_API_URL env var.
        sandbox_id: Reuse an existing sandbox by ID. If None, creates a new one.
        snapshot: Snapshot name/ID for the sandbox.
        sandbox_options: Additional kwargs for CreateSandboxFromSnapshotParams.

    Example::

        from agno.backends.daytona import DaytonaSandbox
        from agno.tools.backend import BackendToolkit
        from agno.agent import Agent

        agent = Agent(tools=[BackendToolkit(DaytonaSandbox(api_key="..."))])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        sandbox_id: Optional[str] = None,
        snapshot: Optional[str] = None,
        sandbox_options: Optional[dict[str, Any]] = None,
    ) -> None:
        from os import getenv
        key = api_key or getenv("DAYTONA_API_KEY")
        url = api_url or getenv("DAYTONA_API_URL")

        config_kwargs: dict[str, Any] = {}
        if key:
            config_kwargs["api_key"] = key
        if url:
            config_kwargs["server_url"] = url

        self._client = Daytona(DaytonaConfig(**config_kwargs) if config_kwargs else None)

        if sandbox_id:
            self._sandbox = self._client.get(sandbox_id)
        else:
            params_kwargs = dict(sandbox_options or {})
            if snapshot:
                params_kwargs["snapshot"] = snapshot
            params = CreateSandboxFromSnapshotParams(**params_kwargs) if params_kwargs else None
            self._sandbox = self._client.create(params) if params else self._client.create()

        logger.debug(f"DaytonaSandbox: using sandbox {self._sandbox.id}")

    @property
    def id(self) -> str:
        return str(self._sandbox.id)

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """Run a shell command inside the Daytona sandbox."""
        try:
            result = self._sandbox.process.exec(command, timeout=timeout)
            output_parts = []
            if hasattr(result, "result") and result.result:
                output_parts.append(result.result)
            if hasattr(result, "stderr") and result.stderr:
                output_parts.append("\n".join(f"[stderr] {l}" for l in result.stderr.split("\n") if l))
            output = "\n".join(output_parts) if output_parts else "<no output>"
            exit_code = getattr(result, "exit_code", 0) or 0
            return ExecuteResponse(output=output, exit_code=exit_code)
        except Exception as e:
            return ExecuteResponse(output=f"Error executing command: {e}", exit_code=1)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                self._sandbox.fs.upload_file(path, content)
                responses.append(FileUploadResponse(path=path))
            except Exception as e:
                # Fallback: write via execute using heredoc
                try:
                    import base64
                    content_b64 = base64.b64encode(content).decode("ascii")
                    cmd = f"python3 -c \"import base64,os; os.makedirs(os.path.dirname('{path}') or '.', exist_ok=True); open('{path}','wb').write(base64.b64decode('{content_b64}'))\""
                    result = self.execute(cmd)
                    if result.exit_code == 0:
                        responses.append(FileUploadResponse(path=path))
                    else:
                        responses.append(FileUploadResponse(path=path, error=result.output))
                except Exception as e2:
                    responses.append(FileUploadResponse(path=path, error=str(e2)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self._sandbox.fs.download_file(path)
                if isinstance(content, str):
                    content = content.encode("utf-8")
                responses.append(FileDownloadResponse(path=path, content=content))
            except Exception as e:
                # Fallback: read via execute
                try:
                    import base64
                    result = self.execute(f"python3 -c \"import base64; print(base64.b64encode(open('{path}','rb').read()).decode())\"")
                    if result.exit_code == 0:
                        raw = base64.b64decode(result.output.strip())
                        responses.append(FileDownloadResponse(path=path, content=raw))
                    else:
                        responses.append(FileDownloadResponse(path=path, error=result.output))
                except Exception as e2:
                    responses.append(FileDownloadResponse(path=path, error=str(e2)))
        return responses

    def close(self) -> None:
        """Remove the Daytona sandbox."""
        try:
            self._client.delete(self._sandbox)
        except Exception:
            pass
