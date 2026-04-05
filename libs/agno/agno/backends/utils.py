"""Shared utilities for backend implementations."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_INLINE_EDIT_BYTES: int = 50 * 1024
"""Max combined size of old_text + new_text for inline edit (50 KB).
Payloads above this use an upload-based edit path to avoid sandbox
request body size limits.
"""

MAX_READ_BYTES: int = 500 * 1024
"""Maximum bytes returned from a single read call (500 KB)."""

MAX_OUTPUT_BYTES: int = 1_000_000
"""Maximum bytes captured from execute() stdout+stderr (1 MB)."""

TRUNCATION_NOTICE = "\n\n[Output truncated due to size limits. Use offset/limit to paginate.]"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def truncate_output(text: str, max_bytes: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    """Truncate text to max_bytes (UTF-8 encoded).

    Returns:
        (truncated_text, was_truncated)
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + TRUNCATION_NOTICE, True


def format_stderr(stderr: str) -> str:
    """Prefix each line of stderr with '[stderr] ' for clear attribution."""
    if not stderr:
        return ""
    lines = stderr.rstrip("\n").split("\n")
    return "\n".join(f"[stderr] {line}" for line in lines)


def combine_output(stdout: str, stderr: str) -> str:
    """Combine stdout and formatted stderr into a single output string."""
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(format_stderr(stderr))
    return "\n".join(parts) if parts else "<no output>"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_virtual_path(root: Path, user_path: str) -> Path:
    """Resolve a user-provided path safely under root (virtual_mode).

    Blocks '..' traversal and '~'. Ensures the resolved path is within root.

    Args:
        root: The root directory.
        user_path: Path provided by the caller.

    Returns:
        Resolved absolute Path under root.

    Raises:
        ValueError: If the path attempts to escape the root.
    """
    # Treat all paths as relative to root
    if ".." in user_path or user_path.startswith("~"):
        raise ValueError(f"Path traversal not allowed: {user_path!r}")

    # Strip leading slash so we can join with root
    clean = user_path.lstrip("/\\")
    full = (root / clean).resolve()

    try:
        full.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"Path {full!r} is outside root directory {root!r}") from None

    return full


# ---------------------------------------------------------------------------
# String replacement helper
# ---------------------------------------------------------------------------


def perform_string_replacement(
    content: str,
    old_text: str,
    new_text: str,
    replace_all: bool,
) -> tuple[str, int] | str:
    """Perform string replacement, returning (new_content, count) or error string."""
    count = content.count(old_text)
    if count == 0:
        return f"Error: String not found in file: {old_text!r}"
    if count > 1 and not replace_all:
        return (
            f"Error: String appears {count} times. Use replace_all=True to replace all occurrences."
        )
    new_content = content.replace(old_text, new_text) if replace_all else content.replace(old_text, new_text, 1)
    return new_content, count


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh",
    ".bash", ".zsh", ".fish", ".env", ".gitignore", ".dockerfile",
    ".xml", ".svg", ".csv", ".rst", ".tex", ".r", ".rb", ".go", ".rs",
    ".java", ".kt", ".swift", ".c", ".cpp", ".h", ".hpp", ".cs", ".php",
    ".sql", ".graphql", ".proto", ".tf", ".hcl", ".makefile", ".mk",
}


def is_text_file(path: str) -> bool:
    """Guess whether a file is text based on extension."""
    ext = Path(path).suffix.lower()
    return ext in _TEXT_EXTENSIONS or ext == ""


# ---------------------------------------------------------------------------
# Base64 helpers for sandbox scripts
# ---------------------------------------------------------------------------


def b64_encode(s: str) -> str:
    """Base64-encode a UTF-8 string, return ASCII string."""
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def b64_decode(s: str) -> str:
    """Base64-decode an ASCII string to UTF-8 string."""
    return base64.b64decode(s.encode("ascii")).decode("utf-8")
