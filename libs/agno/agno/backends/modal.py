"""ModalSandbox: BaseSandbox implementation using Modal cloud containers."""

from __future__ import annotations

from typing import Any, Optional

try:
    import modal
except ImportError:
    raise ImportError(
        "`modal` not installed. Install with: pip install modal"
    )

from agno.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from agno.backends.sandbox import BaseSandbox
from agno.utils.log import logger


class ModalSandbox(BaseSandbox):
    """Sandbox backend backed by Modal cloud sandboxes.

    Wraps a modal.Sandbox object. All file operations are inherited from
    BaseSandbox and run via execute() inside the Modal sandbox.

    Args:
        sandbox: An existing modal.Sandbox instance. If None, one is created
            using app_name and image.
        app_name: Modal app name (used when sandbox=None).
        image: Modal image to use (used when sandbox=None).
        timeout: Command execution timeout in seconds. Default: 1800.

    Example::

        import modal
        from agno.backends.modal import ModalSandbox
        from agno.tools.backend import BackendToolkit
        from agno.agent import Agent

        sb = modal.Sandbox.create(app=modal.App("my-app"))
        agent = Agent(tools=[BackendToolkit(ModalSandbox(sandbox=sb))])
    """

    def __init__(
        self,
        sandbox: Optional[Any] = None,
        app_name: Optional[str] = None,
        image: Optional[Any] = None,
        timeout: int = 1800,
    ) -> None:
        if sandbox is not None:
            self._sandbox = sandbox
        else:
            app = modal.App(app_name or "agno-sandbox")
            img = image or modal.Image.debian_slim()
            self._sandbox = modal.Sandbox.create(app=app, image=img, timeout=timeout)
        self._default_timeout = timeout
        logger.debug(f"ModalSandbox: using sandbox {self._sandbox.object_id}")

    @property
    def id(self) -> str:
        return str(self._sandbox.object_id)

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """Run a shell command in the Modal sandbox."""
        try:
            effective_timeout = timeout or self._default_timeout
            proc = self._sandbox.exec("bash", "-c", command, timeout=effective_timeout)
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            proc.wait()
            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append("\n".join(f"[stderr] {l}" for l in stderr.split("\n") if l))
            output = "\n".join(output_parts) if output_parts else "<no output>"
            return ExecuteResponse(output=output, exit_code=proc.returncode or 0)
        except Exception as e:
            return ExecuteResponse(output=f"Error executing command: {e}", exit_code=1)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                with self._sandbox.open(path, "wb") as f:
                    f.write(content)
                responses.append(FileUploadResponse(path=path))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                with self._sandbox.open(path, "rb") as f:
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content))
            except FileNotFoundError:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            except Exception as e:
                responses.append(FileDownloadResponse(path=path, error=str(e)))
        return responses

    def close(self) -> None:
        """Terminate the Modal sandbox."""
        try:
            self._sandbox.terminate()
        except Exception:
            pass
