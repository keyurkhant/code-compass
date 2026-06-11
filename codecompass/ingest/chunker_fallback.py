from pathlib import Path

from codecompass.ingest.language import detect_language
from codecompass.ingest.models import CodeChunk


class FallbackChunker:
    def __init__(self, window: int = 60, overlap: int = 10):
        self.window = window
        self.overlap = overlap

    def chunk(self, source: str, path: str, repo: str) -> list[CodeChunk]:
        language = detect_language(Path(path)) or "text"
        lines = source.splitlines()
        chunks: list[CodeChunk] = []

        step = self.window - self.overlap
        for start in range(0, max(1, len(lines)), step):
            end = min(start + self.window, len(lines))
            content = "\n".join(lines[start:end])
            if not content.strip():
                continue

            context_prefix = f"# File: {path}"
            chunk_id = CodeChunk.make_id(repo, path, start + 1)
            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    repo=repo,
                    path=path,
                    language=language,
                    symbol_name=None,
                    start_line=start + 1,
                    end_line=end,
                    content=content,
                    context_prefix=context_prefix,
                )
            )

            if end >= len(lines):
                break

        return chunks
