import hashlib
from dataclasses import dataclass


@dataclass
class CodeChunk:
    id: str
    repo: str
    path: str  # relative to repo root
    language: str
    symbol_name: str | None
    start_line: int
    end_line: int
    content: str  # raw source text of the chunk
    context_prefix: str  # file path header + enclosing signature + key imports

    @staticmethod
    def make_id(repo: str, path: str, start_line: int) -> str:
        key = f"{repo}:{path}:{start_line}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def embed_text(self) -> str:
        """Text used for embedding: context prefix + content."""
        return f"{self.context_prefix}\n{self.content}" if self.context_prefix else self.content
