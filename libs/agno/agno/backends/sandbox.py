"""BaseSandbox: implements all file ops via execute().

Concrete subclasses only need to implement execute(), upload_files(),
download_files(), and the id property. All higher-level file operations
(ls, read, write, edit, grep, glob) are derived automatically by running
shell/Python commands through execute().
"""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from typing import Optional

from agno.backends.protocol import (
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
)
from agno.backends.utils import (
    MAX_INLINE_EDIT_BYTES,
    b64_encode,
    is_text_file,
)

# ---------------------------------------------------------------------------
# Server-side script templates
# All parameters are base64-encoded to avoid shell escaping issues.
# ---------------------------------------------------------------------------

_LS_SCRIPT = """python3 -c "
import os, json, base64
path = base64.b64decode('{path_b64}').decode('utf-8')
try:
    entries = []
    with os.scandir(path) as it:
        for e in it:
            try:
                stat = e.stat(follow_symlinks=False)
                entries.append(json.dumps({{'path': os.path.join(path, e.name), 'is_dir': e.is_dir(follow_symlinks=False), 'size': stat.st_size}}))
            except OSError:
                entries.append(json.dumps({{'path': os.path.join(path, e.name), 'is_dir': False}}))
    for entry in sorted(entries):
        print(entry)
except (FileNotFoundError, PermissionError) as ex:
    print(json.dumps({{'error': str(ex)}}))
" 2>&1"""

_READ_SCRIPT = """python3 -c "
import os, sys, base64, json

MAX_BYTES = 500 * 1024
TRUNCATION = '\\n\\n[Output truncated. Use a larger offset or smaller limit to read the rest.]'

path = base64.b64decode('{path_b64}').decode('utf-8')
file_type = '{file_type}'
offset = {offset}
limit = {limit}

if not os.path.isfile(path):
    print(json.dumps({{'error': 'File not found: ' + path}}))
    sys.exit(0)

if os.path.getsize(path) == 0:
    print(json.dumps({{'encoding': 'utf-8', 'content': ''}}))
    sys.exit(0)

if file_type != 'text':
    import base64 as b64
    with open(path, 'rb') as f:
        raw = f.read()
    print(json.dumps({{'encoding': 'base64', 'content': b64.b64encode(raw).decode('ascii')}}))
    sys.exit(0)

try:
    with open(path, 'r', encoding='utf-8', newline=None) as f:
        lines = f.readlines()
except UnicodeDecodeError:
    import base64 as b64
    with open(path, 'rb') as f:
        raw = f.read()
    print(json.dumps({{'encoding': 'base64', 'content': b64.b64encode(raw).decode('ascii')}}))
    sys.exit(0)

if offset >= len(lines):
    print(json.dumps({{'error': 'Offset ' + str(offset) + ' exceeds file length (' + str(len(lines)) + ' lines)'}}))
    sys.exit(0)

selected = lines[offset:offset + limit]
text = ''.join(selected).rstrip('\\n')
encoded = text.encode('utf-8')
truncated = False
if len(encoded) > MAX_BYTES:
    text = encoded[:MAX_BYTES].decode('utf-8', errors='ignore') + TRUNCATION
    truncated = True

print(json.dumps({{'encoding': 'utf-8', 'content': text, 'truncated': truncated}}))
" 2>&1"""

_WRITE_CHECK_SCRIPT = """python3 -c "
import os, sys, base64
path = base64.b64decode('{path_b64}').decode('utf-8')
if os.path.exists(path):
    print('Error: File already exists: ' + repr(path))
    sys.exit(1)
os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
" 2>&1"""

_EDIT_INLINE_SCRIPT = """python3 << '__AGNO_EOF__'
import sys, os, json, base64
data = json.loads(base64.b64decode('{payload_b64}').decode('utf-8'))
path, old, new, replace_all = data['path'], data['old'], data['new'], data['replace_all']
if not os.path.isfile(path):
    print(json.dumps({{'error': 'file_not_found'}}))
    sys.exit(0)
try:
    text = open(path, 'r', encoding='utf-8').read()
except UnicodeDecodeError:
    print(json.dumps({{'error': 'not_a_text_file'}}))
    sys.exit(0)
count = text.count(old)
if count == 0:
    print(json.dumps({{'error': 'string_not_found'}}))
    sys.exit(0)
if count > 1 and not replace_all:
    print(json.dumps({{'error': 'multiple_occurrences', 'count': count}}))
    sys.exit(0)
result = text.replace(old, new) if replace_all else text.replace(old, new, 1)
open(path, 'w', encoding='utf-8').write(result)
print(json.dumps({{'count': count}}))
__AGNO_EOF__
"""

_EDIT_TMPFILE_SCRIPT = """python3 -c "
import os, sys, json, base64
old_path = base64.b64decode('{old_path_b64}').decode('utf-8')
new_path = base64.b64decode('{new_path_b64}').decode('utf-8')
target = base64.b64decode('{target_b64}').decode('utf-8')
replace_all = {replace_all}
try:
    old = open(old_path, 'rb').read().decode('utf-8')
    new = open(new_path, 'rb').read().decode('utf-8')
except Exception as e:
    print(json.dumps({{'error': 'temp_read_failed', 'detail': str(e)}}))
    sys.exit(0)
finally:
    for p in (old_path, new_path):
        try: os.remove(p)
        except OSError: pass
if not os.path.isfile(target):
    print(json.dumps({{'error': 'file_not_found'}}))
    sys.exit(0)
try:
    text = open(target, 'r', encoding='utf-8').read()
except UnicodeDecodeError:
    print(json.dumps({{'error': 'not_a_text_file'}}))
    sys.exit(0)
count = text.count(old)
if count == 0:
    print(json.dumps({{'error': 'string_not_found'}}))
    sys.exit(0)
if count > 1 and not replace_all:
    print(json.dumps({{'error': 'multiple_occurrences', 'count': count}}))
    sys.exit(0)
result = text.replace(old, new) if replace_all else text.replace(old, new, 1)
open(target, 'w', encoding='utf-8').write(result)
print(json.dumps({{'count': count}}))
" 2>&1"""

_GLOB_SCRIPT = """python3 -c "
import glob as g, os, json, base64
path = base64.b64decode('{path_b64}').decode('utf-8')
pattern = base64.b64decode('{pattern_b64}').decode('utf-8')
os.chdir(path)
for m in sorted(g.glob(pattern, recursive=True)):
    print(json.dumps({{'path': m, 'is_dir': os.path.isdir(m)}}))
" 2>&1"""


class BaseSandbox(SandboxBackendProtocol):
    """Base class for sandbox backends.

    Implements all file operations (ls, read, write, edit, grep, glob) by
    delegating to execute(). Subclasses only need to implement:
    - execute(command, *, timeout) -> ExecuteResponse
    - upload_files(files) -> list[FileUploadResponse]
    - download_files(paths) -> list[FileDownloadResponse]
    - id property

    This means adding a new sandbox provider only requires wrapping its
    SDK in those four methods — all file ops come for free.
    """

    # -- Abstract interface --------------------------------------------------

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this sandbox instance."""

    @abstractmethod
    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """Execute a shell command in the sandbox."""

    @abstractmethod
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox. Must handle partial success per-file."""

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox. Must handle partial success per-file."""

    # -- File ops implemented via execute() ----------------------------------

    def ls(self, path: str = ".") -> LsResult:
        path_b64 = b64_encode(path)
        result = self.execute(_LS_SCRIPT.format(path_b64=path_b64))
        entries: list[LsEntry] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if "error" in data:
                    return LsResult(error=data["error"])
                entries.append(LsEntry(
                    path=data["path"],
                    is_dir=data.get("is_dir", False),
                    size=data.get("size"),
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        return LsResult(entries=entries)

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        path_b64 = b64_encode(path)
        file_type = "text" if is_text_file(path) else "binary"
        cmd = _READ_SCRIPT.format(
            path_b64=path_b64,
            file_type=file_type,
            offset=int(offset),
            limit=int(limit),
        )
        result = self.execute(cmd)
        output = result.output.rstrip()
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return ReadResult(error=f"Unexpected server response for '{path}': {output[:200]}")
        if "error" in data:
            return ReadResult(error=data["error"])
        return ReadResult(
            content=data.get("content", ""),
            encoding=data.get("encoding", "utf-8"),
            truncated=data.get("truncated", False),
        )

    def write(self, path: str, content: str) -> WriteResult:
        path_b64 = b64_encode(path)
        check_cmd = _WRITE_CHECK_SCRIPT.format(path_b64=path_b64)
        check = self.execute(check_cmd)
        if check.exit_code != 0 or "Error:" in check.output:
            return WriteResult(error=check.output.strip() or f"Failed to write '{path}'")
        responses = self.upload_files([(path, content.encode("utf-8"))])
        if not responses:
            return WriteResult(error=f"upload_files returned no response for '{path}'")
        if responses[0].error:
            return WriteResult(error=f"Failed to upload '{path}': {responses[0].error}")
        return WriteResult(path=path)

    def edit(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> EditResult:
        payload_size = len(old_text.encode("utf-8")) + len(new_text.encode("utf-8"))
        if payload_size <= MAX_INLINE_EDIT_BYTES:
            return self._edit_inline(path, old_text, new_text, replace_all)
        return self._edit_via_upload(path, old_text, new_text, replace_all)

    def _edit_inline(self, path: str, old_text: str, new_text: str, replace_all: bool) -> EditResult:
        import json as _json
        payload = _json.dumps({"path": path, "old": old_text, "new": new_text, "replace_all": replace_all})
        payload_b64 = b64_encode(payload)
        cmd = _EDIT_INLINE_SCRIPT.format(payload_b64=payload_b64)
        result = self.execute(cmd)
        return self._parse_edit_result(result.output.rstrip(), path, old_text)

    def _edit_via_upload(self, path: str, old_text: str, new_text: str, replace_all: bool) -> EditResult:
        uid = b64_encode(os.urandom(10).hex())[:12]
        old_tmp = f"/tmp/.agno_edit_{uid}_old"
        new_tmp = f"/tmp/.agno_edit_{uid}_new"
        resps = self.upload_files([
            (old_tmp, old_text.encode("utf-8")),
            (new_tmp, new_text.encode("utf-8")),
        ])
        if len(resps) < 2 or any(r.error for r in resps):
            errors = [r.error for r in resps if r.error]
            return EditResult(error=f"Failed to upload temp files for edit: {errors}")
        cmd = _EDIT_TMPFILE_SCRIPT.format(
            old_path_b64=b64_encode(old_tmp),
            new_path_b64=b64_encode(new_tmp),
            target_b64=b64_encode(path),
            replace_all=str(replace_all),
        )
        result = self.execute(cmd)
        return self._parse_edit_result(result.output.rstrip(), path, old_text)

    @staticmethod
    def _parse_edit_result(output: str, path: str, old_text: str) -> EditResult:
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return EditResult(error=f"Unexpected edit response for '{path}': {output[:200]}")
        if "error" in data:
            err = data["error"]
            if err == "file_not_found":
                return EditResult(error=f"File not found: '{path}'")
            if err == "not_a_text_file":
                return EditResult(error=f"File '{path}' is not a text file")
            if err == "string_not_found":
                return EditResult(error=f"String not found in '{path}': {old_text!r}")
            if err == "multiple_occurrences":
                return EditResult(error=f"String appears {data.get('count', '?')} times in '{path}'. Use replace_all=True.")
            return EditResult(error=f"Edit error for '{path}': {err}")
        return EditResult(path=path, occurrences=data.get("count", 1))

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        import shlex
        search_path = shlex.quote(path or ".")
        pattern_escaped = shlex.quote(pattern)
        glob_flag = f"--include={shlex.quote(glob)}" if glob else ""
        cmd = f"grep -rHnF {glob_flag} -e {pattern_escaped} {search_path} 2>/dev/null || true"
        result = self.execute(cmd)
        output = result.output.rstrip()
        if not output:
            return GrepResult(matches=[])
        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                try:
                    matches.append(GrepMatch(
                        path=parts[0],
                        line_number=int(parts[1]),
                        line_text=parts[2],
                    ))
                except ValueError:
                    continue
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        base = path or "."
        path_b64 = b64_encode(base)
        pattern_b64 = b64_encode(pattern)
        cmd = _GLOB_SCRIPT.format(path_b64=path_b64, pattern_b64=pattern_b64)
        result = self.execute(cmd)
        paths: list[str] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if not data.get("is_dir", False):
                    paths.append(data["path"])
            except (json.JSONDecodeError, KeyError):
                continue
        return GlobResult(paths=paths)
