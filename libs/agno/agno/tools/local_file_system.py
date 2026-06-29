from pathlib import Path
from typing import Optional
from uuid import uuid4

from agno.tools import Toolkit
from agno.utils.log import log_debug, log_error


class LocalFileSystemTools(Toolkit):
    def __init__(
        self,
        target_directory: Optional[str] = None,
        default_extension: str = "txt",
        enable_write_file: bool = True,
        all: bool = False,
        **kwargs,
    ):
        """
        Initialize the WriteToLocal toolkit.
        Args:
            target_directory (Optional[str]): Default directory to write files to. Creates if doesn't exist.
            default_extension (str): Default file extension to use if none specified.
        """

        self.target_directory = target_directory or str(Path.cwd())
        self.default_extension = default_extension.lstrip(".")

        target_path = Path(self.target_directory)
        target_path.mkdir(parents=True, exist_ok=True)

        tools = []
        if all or enable_write_file:
            tools.append(self.write_file)

        super().__init__(name="write_to_local", tools=tools, **kwargs)

    def _resolve_within_target(self, candidate: Path) -> Optional[Path]:
        """Resolve ``candidate`` and confine it to ``target_directory``.

        Returns the fully resolved path when it stays inside the target
        directory, or ``None`` when it would escape (path traversal). The
        check is version-safe (no ``Path.is_relative_to``, which is 3.9+).
        """
        target_base = Path(self.target_directory).resolve()
        resolved = candidate.resolve()
        if resolved == target_base or target_base in resolved.parents:
            return resolved
        return None

    def write_file(
        self,
        content: str,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
        extension: Optional[str] = None,
    ) -> str:
        """
        Write content to a local file.
        Args:
            content (str): Content to write to the file
            filename (Optional[str]): Name of the file. Defaults to UUID if not provided
            directory (Optional[str]): Directory to write file to. Uses target_directory if not provided
            extension (Optional[str]): File extension. Uses default_extension if not provided
        Returns:
            str: Path to the created file or error message
        """
        try:
            filename = filename or str(uuid4())
            directory = directory or self.target_directory
            if filename and "." in filename:
                path_obj = Path(filename)
                filename = path_obj.stem
                extension = extension or path_obj.suffix.lstrip(".")

            log_debug(f"Writing file to local system: {filename}")

            extension = (extension or self.default_extension).lstrip(".")

            # Construct full filename with extension
            full_filename = f"{filename}.{extension}"
            file_path = Path(directory) / full_filename

            # Security: confine all writes to the target directory (prevent path traversal).
            # Validate before creating any directories so a traversal attempt cannot leave
            # stray directories outside the sandbox.
            resolved_path = self._resolve_within_target(file_path)
            if resolved_path is None:
                error_msg = "Path traversal detected. Cannot write outside the target directory."
                log_error(error_msg)
                return f"Error: {error_msg}"

            # Create directory if it doesn't exist
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            resolved_path.write_text(content)

            return f"Successfully wrote file to: {resolved_path}"

        except Exception as e:
            error_msg = f"Failed to write file: {str(e)}"
            log_error(error_msg)
            return f"Error: {error_msg}"

    def read_file(self, filename: str, directory: Optional[str] = None) -> str:
        """
        Read content from a local file.
        """
        file_path = Path(directory or self.target_directory) / filename

        # Security: confine reads to the target directory (prevent path traversal)
        resolved_path = self._resolve_within_target(file_path)
        if resolved_path is None:
            return "Error: Path traversal detected. Cannot read outside the target directory."

        if not resolved_path.exists():
            return f"File not found: {resolved_path}"
        return resolved_path.read_text()
