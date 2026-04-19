"""FilesystemBackend: read/write files directly on the local filesystem.

Security note: Use virtual_mode=True to restrict all paths to a root
directory. In virtual_mode=False (default), agents have unrestricted
access to the filesystem.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from agno.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsEntry,
    LsResult,
    ReadResult,
    WriteResult,
)
from agno.backends.utils import (
    is_text_file,
    perform_string_replacement,
    resolve_virtual_path,
)


class FilesystemBackend(BackendProtocol):
    """Backend that reads and writes files directly from the local filesystem.

    Args:
        root: Root directory for file operations. Defaults to current working directory.
        virtual_mode: If True, all paths are treated as relative to ``root``
            and path traversal (``..``, ``~``) is blocked. If False (default),
            absolute paths are used as-is.
        max_file_size_mb: Maximum file size for grep Python fallback (default 10 MB).

    .. warning::
        This backend grants agents direct filesystem read/write access.
        Use virtual_mode=True and restrict root to a safe directory when
        running in untrusted contexts.
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        virtual_mode: bool = False,
        max_file_size_mb: int = 10,
    ) -> None:
        self.root = Path(root).resolve() if root else Path.cwd()
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    # -- Path resolution -----------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Resolve a user-provided path to an absolute filesystem path."""
        if self.virtual_mode:
            return resolve_virtual_path(self.root, path)
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.root / p).resolve()

    def _to_display_path(self, abs_path: Path) -> str:
        """Convert an absolute path to a display path (virtual or real)."""
        if self.virtual_mode:
            try:
                rel = abs_path.resolve().relative_to(self.root.resolve())
                return "/" + rel.as_posix()
            except ValueError:
                return str(abs_path)
        return str(abs_path)

    # -- ls ------------------------------------------------------------------

    def ls(self, path: str = ".") -> LsResult:
        try:
            dir_path = self._resolve(path)
        except ValueError as e:
            return LsResult(error=str(e))

        if not dir_path.exists() or not dir_path.is_dir():
            return LsResult(entries=[], error=None)

        entries: list[LsEntry] = []
        try:
            for child in sorted(dir_path.iterdir(), key=lambda p: p.name):
                try:
                    is_dir = child.is_dir()
                    stat = child.stat()
                    display = self._to_display_path(child)
                    entries.append(LsEntry(
                        path=display + ("/" if is_dir else ""),
                        is_dir=is_dir,
                        size=stat.st_size if not is_dir else 0,
                        modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    ))
                except OSError:
                    continue
        except (OSError, PermissionError):
            pass

        return LsResult(entries=entries)

    # -- read ----------------------------------------------------------------

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        try:
            resolved = self._resolve(path)
        except ValueError as e:
            return ReadResult(error=str(e))

        if not resolved.exists() or not resolved.is_file():
            return ReadResult(error=f"File not found: '{path}'")

        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW

            if not is_text_file(path):
                fd = os.open(resolved, flags)
                with os.fdopen(fd, "rb") as f:
                    raw = f.read()
                encoded = base64.standard_b64encode(raw).decode("ascii")
                return ReadResult(content=encoded, encoding="base64")

            fd = os.open(resolved, flags)
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                return ReadResult(content="", encoding="utf-8")

            if offset >= len(lines):
                return ReadResult(
                    error=f"Offset {offset} exceeds file length ({len(lines)} lines)"
                )

            selected = lines[offset: offset + limit]
            content = "".join(selected).rstrip("\n")
            return ReadResult(content=content, encoding="utf-8")

        except UnicodeDecodeError:
            # Retry as binary
            try:
                with open(resolved, "rb") as f:
                    raw = f.read()
                encoded = base64.standard_b64encode(raw).decode("ascii")
                return ReadResult(content=encoded, encoding="base64")
            except OSError as e:
                return ReadResult(error=f"Error reading '{path}': {e}")
        except OSError as e:
            return ReadResult(error=f"Error reading '{path}': {e}")

    # -- write ---------------------------------------------------------------

    def write(self, path: str, content: str) -> WriteResult:
        try:
            resolved = self._resolve(path)
        except ValueError as e:
            return WriteResult(error=str(e))

        if resolved.exists():
            return WriteResult(
                error=f"File already exists: '{path}'. Read and edit it, or write to a new path."
            )

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            return WriteResult(path=self._to_display_path(resolved))
        except (OSError, UnicodeEncodeError) as e:
            return WriteResult(error=f"Error writing '{path}': {e}")

    # -- edit ----------------------------------------------------------------

    def edit(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> EditResult:
        try:
            resolved = self._resolve(path)
        except ValueError as e:
            return EditResult(error=str(e))

        if not resolved.exists() or not resolved.is_file():
            return EditResult(error=f"File not found: '{path}'")

        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved, flags)
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            # Normalize line endings
            old_text = old_text.replace("\r\n", "\n").replace("\r", "\n")
            new_text = new_text.replace("\r\n", "\n").replace("\r", "\n")

            result = perform_string_replacement(content, old_text, new_text, replace_all)
            if isinstance(result, str):
                return EditResult(error=result)

            new_content, occurrences = result

            flags = os.O_WRONLY | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)

            return EditResult(path=self._to_display_path(resolved), occurrences=int(occurrences))
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error editing '{path}': {e}")

    # -- grep ----------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        try:
            base = self._resolve(path or ".")
        except ValueError as e:
            return GrepResult(error=str(e))

        if not base.exists():
            return GrepResult(matches=[])

        # Try ripgrep first (literal/fixed-string mode)
        result = self._ripgrep(pattern, base, glob)
        if result is not None:
            return result

        # Python fallback
        return self._python_grep(pattern, base, glob)

    def _ripgrep(self, pattern: str, base: Path, include_glob: Optional[str]) -> Optional[GrepResult]:
        cmd = ["rg", "--json", "-F"]
        if include_glob:
            cmd.extend(["--glob", include_glob])
        cmd.extend(["--", pattern, str(base)])
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        import json
        matches: list[GrepMatch] = []
        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except Exception:
                continue
            if data.get("type") != "match":
                continue
            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue
            display = self._to_display_path(Path(ftext))
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            matches.append(GrepMatch(path=display, line_number=int(ln), line_text=lt))
        return GrepResult(matches=matches)

    def _python_grep(self, pattern: str, base: Path, include_glob: Optional[str]) -> GrepResult:
        try:
            regex = re.compile(re.escape(pattern))
        except re.error as e:
            return GrepResult(error=f"Invalid pattern: {e}")

        matches: list[GrepMatch] = []
        root = base if base.is_dir() else base.parent

        for fp in root.rglob("*"):
            try:
                if not fp.is_file():
                    continue
            except (PermissionError, OSError):
                continue

            if include_glob:
                rel = str(fp.relative_to(root))
                import fnmatch
                if not fnmatch.fnmatch(rel, include_glob):
                    continue

            try:
                if fp.stat().st_size > self.max_file_size_bytes:
                    continue
            except OSError:
                continue

            try:
                content = fp.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

            display = self._to_display_path(fp)
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    matches.append(GrepMatch(path=display, line_number=line_num, line_text=line))

        return GrepResult(matches=matches)

    # -- glob ----------------------------------------------------------------

    def glob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        try:
            base = self._resolve(path or ".")
        except ValueError as e:
            return GlobResult(error=str(e))

        if not base.exists() or not base.is_dir():
            return GlobResult(paths=[])

        paths: list[str] = []
        try:
            for matched in sorted(base.rglob(pattern)):
                try:
                    if not matched.is_file():
                        continue
                    if self.virtual_mode:
                        try:
                            matched.resolve().relative_to(self.root.resolve())
                        except ValueError:
                            continue
                    paths.append(self._to_display_path(matched))
                except (PermissionError, OSError):
                    continue
        except (OSError, ValueError):
            pass

        return GlobResult(paths=paths)

    # -- upload / download ---------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                resolved = self._resolve(path)
                resolved.parent.mkdir(parents=True, exist_ok=True)
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(resolved, flags, 0o644)
                with os.fdopen(fd, "wb") as f:
                    f.write(content)
                responses.append(FileUploadResponse(path=path))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                resolved = self._resolve(path)
                flags = os.O_RDONLY
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(resolved, flags)
                with os.fdopen(fd, "rb") as f:
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content))
            except FileNotFoundError:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            except PermissionError:
                responses.append(FileDownloadResponse(path=path, error="permission_denied"))
            except Exception as e:
                responses.append(FileDownloadResponse(path=path, error=str(e)))
        return responses
