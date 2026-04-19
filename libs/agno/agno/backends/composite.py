"""CompositeBackend: route file operations to different backends by path prefix.

Example::

    from agno.backends import FilesystemBackend, CompositeBackend
    from agno.backends.e2b import E2BSandbox

    composite = CompositeBackend(
        default=FilesystemBackend(root="/workspace"),
        routes={
            "/remote/": E2BSandbox(api_key="..."),
        }
    )
    # /workspace files go to FilesystemBackend
    # /remote/* paths go to E2BSandbox
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from agno.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsEntry,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
    execute_accepts_timeout,
)


def _route_for_path(
    default: BackendProtocol,
    sorted_routes: list[tuple[str, BackendProtocol]],
    path: str,
) -> tuple[BackendProtocol, str, Optional[str]]:
    """Find the backend for path, return (backend, backend_path, matched_prefix).

    Strips the route prefix from path when forwarding to the backend.
    Returns (default, original_path, None) if no route matches.
    """
    for prefix, backend in sorted_routes:
        prefix_no_slash = prefix.rstrip("/")
        # Exact match: e.g. path="/remote" prefix="/remote/"
        if path == prefix_no_slash:
            return backend, "/", prefix
        # Prefix match: e.g. path="/remote/foo.py" prefix="/remote/"
        norm_prefix = prefix if prefix.endswith("/") else prefix + "/"
        if path.startswith(norm_prefix):
            suffix = path[len(norm_prefix):]
            backend_path = "/" + suffix if suffix else "/"
            return backend, backend_path, prefix
    return default, path, None


class CompositeBackend(BackendProtocol):
    """Routes file operations to different backends based on path prefixes.

    Routes are matched longest-first, so more specific prefixes take priority.
    Unmatched paths go to the default backend.

    When path is "/" or None for grep/ls, results are aggregated from all backends.
    execute() always uses the default backend (execution is not path-routable).

    Args:
        default: Backend for paths that don't match any route.
        routes: Map of path prefixes to backends (e.g. {"/memories/": store_backend}).
    """

    def __init__(
        self,
        default: BackendProtocol,
        routes: dict[str, BackendProtocol],
    ) -> None:
        self.default = default
        self.routes = routes
        # Sort by prefix length descending — longest match wins
        self.sorted_routes: list[tuple[str, BackendProtocol]] = sorted(
            routes.items(), key=lambda x: len(x[0]), reverse=True
        )

    def _route(self, path: str) -> tuple[BackendProtocol, str, Optional[str]]:
        return _route_for_path(self.default, self.sorted_routes, path)

    def _backend_for(self, path: str) -> tuple[BackendProtocol, str]:
        backend, backend_path, _ = self._route(path)
        return backend, backend_path

    # -- ls ------------------------------------------------------------------

    def ls(self, path: str = ".") -> LsResult:
        backend, backend_path, prefix = self._route(path)

        if prefix is not None:
            result = backend.ls(backend_path)
            if result.error:
                return result
            return LsResult(entries=[
                LsEntry(
                    path=f"{prefix.rstrip('/')}{e.path}",
                    is_dir=e.is_dir,
                    size=e.size,
                    modified_at=e.modified_at,
                )
                for e in (result.entries or [])
            ])

        # Root: aggregate default + synthetic route dirs
        if path in ("/", "."):
            entries: list[LsEntry] = []
            default_result = self.default.ls(path)
            entries.extend(default_result.entries or [])
            for route_prefix in self.routes:
                entries.append(LsEntry(path=route_prefix, is_dir=True, size=0))
            entries.sort(key=lambda e: e.path)
            return LsResult(entries=entries)

        return self.default.ls(path)

    async def als(self, path: str = ".") -> LsResult:
        backend, backend_path, prefix = self._route(path)
        if prefix is not None:
            result = await backend.als(backend_path)
            if result.error:
                return result
            return LsResult(entries=[
                LsEntry(path=f"{prefix.rstrip('/')}{e.path}", is_dir=e.is_dir, size=e.size, modified_at=e.modified_at)
                for e in (result.entries or [])
            ])
        if path in ("/", "."):
            entries = []
            default_result = await self.default.als(path)
            entries.extend(default_result.entries or [])
            for route_prefix in self.routes:
                entries.append(LsEntry(path=route_prefix, is_dir=True, size=0))
            entries.sort(key=lambda e: e.path)
            return LsResult(entries=entries)
        return await self.default.als(path)

    # -- read/write/edit -----------------------------------------------------

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        backend, backend_path = self._backend_for(path)
        return backend.read(backend_path, offset, limit)

    async def aread(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        backend, backend_path = self._backend_for(path)
        return await backend.aread(backend_path, offset, limit)

    def write(self, path: str, content: str) -> WriteResult:
        backend, backend_path = self._backend_for(path)
        result = backend.write(backend_path, content)
        if result.path is not None:
            return WriteResult(path=path)
        return result

    async def awrite(self, path: str, content: str) -> WriteResult:
        backend, backend_path = self._backend_for(path)
        result = await backend.awrite(backend_path, content)
        if result.path is not None:
            return WriteResult(path=path)
        return result

    def edit(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> EditResult:
        backend, backend_path = self._backend_for(path)
        result = backend.edit(backend_path, old_text, new_text, replace_all)
        if result.path is not None:
            return EditResult(path=path, occurrences=result.occurrences)
        return result

    async def aedit(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> EditResult:
        backend, backend_path = self._backend_for(path)
        result = await backend.aedit(backend_path, old_text, new_text, replace_all)
        if result.path is not None:
            return EditResult(path=path, occurrences=result.occurrences)
        return result

    # -- grep ----------------------------------------------------------------

    def grep(self, pattern: str, path: Optional[str] = None, glob: Optional[str] = None) -> GrepResult:
        if path is not None:
            backend, backend_path, prefix = self._route(path)
            if prefix is not None:
                result = backend.grep(pattern, backend_path, glob)
                if result.error:
                    return result
                return GrepResult(matches=[
                    GrepMatch(path=f"{prefix.rstrip('/')}{m.path}", line_number=m.line_number, line_text=m.line_text)
                    for m in (result.matches or [])
                ])

        # Fan out to all backends
        if path is None or path in ("/", "."):
            all_matches: list[GrepMatch] = []
            default_result = self.default.grep(pattern, path, glob)
            if default_result.error:
                return default_result
            all_matches.extend(default_result.matches or [])
            for route_prefix, backend in self.routes.items():
                r = backend.grep(pattern, "/", glob)
                if r.error:
                    return r
                all_matches.extend(
                    GrepMatch(path=f"{route_prefix.rstrip('/')}{m.path}", line_number=m.line_number, line_text=m.line_text)
                    for m in (r.matches or [])
                )
            return GrepResult(matches=all_matches)

        return self.default.grep(pattern, path, glob)

    async def agrep(self, pattern: str, path: Optional[str] = None, glob: Optional[str] = None) -> GrepResult:
        if path is not None:
            backend, backend_path, prefix = self._route(path)
            if prefix is not None:
                result = await backend.agrep(pattern, backend_path, glob)
                if result.error:
                    return result
                return GrepResult(matches=[
                    GrepMatch(path=f"{prefix.rstrip('/')}{m.path}", line_number=m.line_number, line_text=m.line_text)
                    for m in (result.matches or [])
                ])
        if path is None or path in ("/", "."):
            all_matches = []
            default_result = await self.default.agrep(pattern, path, glob)
            if default_result.error:
                return default_result
            all_matches.extend(default_result.matches or [])
            for route_prefix, backend in self.routes.items():
                r = await backend.agrep(pattern, "/", glob)
                if r.error:
                    return r
                all_matches.extend(
                    GrepMatch(path=f"{route_prefix.rstrip('/')}{m.path}", line_number=m.line_number, line_text=m.line_text)
                    for m in (r.matches or [])
                )
            return GrepResult(matches=all_matches)
        return await self.default.agrep(pattern, path, glob)

    # -- glob ----------------------------------------------------------------

    def glob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        backend, backend_path, prefix = self._route(path or ".")
        if prefix is not None:
            result = backend.glob(pattern, backend_path)
            if result.error:
                return result
            return GlobResult(paths=[f"{prefix.rstrip('/')}{p}" for p in result.paths])

        # Fan out to default + all routes
        all_paths: list[str] = []
        default_result = self.default.glob(pattern, path)
        all_paths.extend(default_result.paths or [])
        for route_prefix, backend in self.routes.items():
            r = backend.glob(pattern, "/")
            all_paths.extend(f"{route_prefix.rstrip('/')}{p}" for p in (r.paths or []))
        return GlobResult(paths=sorted(all_paths))

    async def aglob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        backend, backend_path, prefix = self._route(path or ".")
        if prefix is not None:
            result = await backend.aglob(pattern, backend_path)
            if result.error:
                return result
            return GlobResult(paths=[f"{prefix.rstrip('/')}{p}" for p in result.paths])
        all_paths: list[str] = []
        default_result = await self.default.aglob(pattern, path)
        all_paths.extend(default_result.paths or [])
        for route_prefix, backend in self.routes.items():
            r = await backend.aglob(pattern, "/")
            all_paths.extend(f"{route_prefix.rstrip('/')}{p}" for p in (r.paths or []))
        return GlobResult(paths=sorted(all_paths))

    # -- upload / download ---------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        results: list[Optional[FileUploadResponse]] = [None] * len(files)
        batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)
        for idx, (path, content) in enumerate(files):
            backend, stripped = self._backend_for(path)
            batches[backend].append((idx, stripped, content))
        for backend, batch in batches.items():
            indices, stripped_paths, contents = zip(*batch)
            responses = backend.upload_files(list(zip(stripped_paths, contents)))
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],
                    error=responses[i].error if i < len(responses) else None,
                )
        return results  # type: ignore[return-value]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results: list[Optional[FileDownloadResponse]] = [None] * len(paths)
        batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)
        for idx, path in enumerate(paths):
            backend, stripped = self._backend_for(path)
            batches[backend].append((idx, stripped))
        for backend, batch in batches.items():
            indices, stripped_paths = zip(*batch)
            responses = backend.download_files(list(stripped_paths))
            for i, orig_idx in enumerate(indices):
                resp = responses[i] if i < len(responses) else None
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],
                    content=resp.content if resp else None,
                    error=resp.error if resp else "no response",
                )
        return results  # type: ignore[return-value]

    # -- execute (always to default) -----------------------------------------

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """Execute via the default backend. Execution is not path-routable."""
        if not isinstance(self.default, SandboxBackendProtocol):
            raise NotImplementedError(
                "Default backend does not support execute(). "
                "Use a SandboxBackendProtocol as the default backend."
            )
        if timeout is not None and execute_accepts_timeout(type(self.default)):
            return self.default.execute(command, timeout=timeout)
        return self.default.execute(command)

    async def aexecute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        if not isinstance(self.default, SandboxBackendProtocol):
            raise NotImplementedError(
                "Default backend does not support execute(). "
                "Use a SandboxBackendProtocol as the default backend."
            )
        if timeout is not None and execute_accepts_timeout(type(self.default)):
            return await self.default.aexecute(command, timeout=timeout)
        return await self.default.aexecute(command)
