from agno.tools.local_file_system import LocalFileSystemTools


def test_write_file_within_target_directory(tmp_path):
    """A normal write lands inside the target directory."""
    target = tmp_path / "workspace"
    tools = LocalFileSystemTools(target_directory=str(target))

    result = tools.write_file(content="hello", filename="note.txt")

    assert "Successfully wrote" in result
    assert (target / "note.txt").read_text() == "hello"


def test_write_file_blocks_absolute_directory_escape(tmp_path):
    """write_file must not escape the target directory via the `directory` arg."""
    target = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    tools = LocalFileSystemTools(target_directory=str(target))

    result = tools.write_file(content="hacked", filename="evil.txt", directory=str(outside))

    assert result.lower().startswith("error")
    assert not (outside / "evil.txt").exists()


def test_write_file_blocks_relative_traversal(tmp_path):
    """A `../` directory traversal must be refused and write nothing outside."""
    target = tmp_path / "workspace"
    target.mkdir(parents=True)
    tools = LocalFileSystemTools(target_directory=str(target))

    result = tools.write_file(content="hacked", filename="pwned.txt", directory=str(target / ".."))

    assert result.lower().startswith("error")
    assert not (tmp_path / "pwned.txt").exists()


def test_read_file_blocks_traversal(tmp_path):
    """read_file must not read files outside the target directory."""
    target = tmp_path / "workspace"
    target.mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("top-secret")
    tools = LocalFileSystemTools(target_directory=str(target))

    result = tools.read_file(filename="../secret.txt")

    assert "top-secret" not in result
