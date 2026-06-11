"""Fast local embeddings via fastembed (Qdrant's ONNX-based library).

~10× faster than sentence-transformers on CPU; no GPU required.
Models are downloaded once to ~/.cache/fastembed/.

Good code-aware models:
    jinaai/jina-embeddings-v2-base-code   768-dim  best for code (default)
    BAAI/bge-small-en-v1.5               384-dim  fastest, still good
    BAAI/bge-base-en-v1.5               768-dim  balanced
"""

from __future__ import annotations

from codecompass.providers.base import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code") -> None:
        self._model_name = model_name
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return [emb.tolist() for emb in model.embed(texts)]

    @property
    def dimension(self) -> int:
        if self._dim is None:
            sample = list(self._load().embed(["x"]))
            self._dim = len(sample[0])
        return self._dim
