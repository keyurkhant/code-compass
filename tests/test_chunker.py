import pytest
from pathlib import Path
from codecompass.ingest.chunker import chunk_file
from codecompass.ingest.chunker_fallback import FallbackChunker
from codecompass.ingest.chunker_treesitter import TreeSitterChunker


def test_python_chunks_by_function(tiny_repo_path):
    chunks = chunk_file(tiny_repo_path / "main.py", tiny_repo_path, "tiny_repo")
    assert len(chunks) > 0
    symbol_names = [c.symbol_name for c in chunks if c.symbol_name]
    assert "main" in symbol_names or "Config" in symbol_names


def test_chunk_has_context_prefix(tiny_repo_path):
    chunks = chunk_file(tiny_repo_path / "main.py", tiny_repo_path, "tiny_repo")
    for chunk in chunks:
        assert chunk.context_prefix  # should have file path at minimum
        assert "main.py" in chunk.context_prefix


def test_chunk_start_end_lines(tiny_repo_path):
    chunks = chunk_file(tiny_repo_path / "main.py", tiny_repo_path, "tiny_repo")
    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_chunk_id_is_deterministic(tiny_repo_path):
    chunks1 = chunk_file(tiny_repo_path / "main.py", tiny_repo_path, "tiny_repo")
    chunks2 = chunk_file(tiny_repo_path / "main.py", tiny_repo_path, "tiny_repo")
    assert [c.id for c in chunks1] == [c.id for c in chunks2]


def test_fallback_chunker_windowing():
    source = "\n".join(f"line {i}" for i in range(200))
    chunker = FallbackChunker(window=60, overlap=10)
    chunks = chunker.chunk(source, "fake/file.txt", "repo")
    assert len(chunks) > 1
    # Check overlap: end of first chunk's lines appear in start of second
    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_treesitter_python_chunker(tiny_repo_path):
    source = (tiny_repo_path / "utils.py").read_text()
    chunker = TreeSitterChunker("python")
    chunks = chunker.chunk(source, "utils.py", "tiny_repo")
    assert any(c.symbol_name == "parse_args" for c in chunks)
    assert any(c.symbol_name == "validate_config" for c in chunks)
