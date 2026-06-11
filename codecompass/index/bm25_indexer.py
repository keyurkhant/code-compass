import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from codecompass.ingest.models import CodeChunk


def _tokenize(text: str) -> list[str]:
    """Tokenize preserving code identifiers as single tokens."""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._paths: list[str] = []

    def build(self, chunks: list[CodeChunk]) -> None:
        """Build BM25 index from chunks."""
        corpus = [_tokenize(c.embed_text()) for c in chunks]
        self._chunk_ids = [c.id for c in chunks]
        self._paths = [c.path for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, score) pairs for top_k results."""
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._chunk_ids[i], float(s)) for i, s in ranked[:top_k] if s > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "ids": self._chunk_ids, "paths": self._paths}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["ids"]
        self._paths = data["paths"]

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None
