from pathlib import Path
from codecompass.ingest.models import CodeChunk
from codecompass.ingest.language import detect_language, TREESITTER_SUPPORTED
from codecompass.ingest.chunker_treesitter import TreeSitterChunker
from codecompass.ingest.chunker_fallback import FallbackChunker
import logging

logger = logging.getLogger(__name__)

def chunk_file(path: Path, repo_root: Path, repo_name: str) -> list[CodeChunk]:
    """Read a file and return its chunks with metadata."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning(f"Cannot read {path}: {e}")
        return []

    if not source.strip():
        return []

    rel_path = str(path.relative_to(repo_root))
    language = detect_language(path)

    if language in TREESITTER_SUPPORTED:
        try:
            chunker = TreeSitterChunker(language)
            chunks = chunker.chunk(source, rel_path, repo_name)
            if chunks:
                return chunks
        except Exception as e:
            logger.warning(f"Tree-sitter chunking failed for {path}: {e}, falling back")

    return FallbackChunker().chunk(source, rel_path, repo_name)
