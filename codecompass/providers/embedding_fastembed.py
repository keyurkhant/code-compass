"""Fast local embeddings via fastembed (Qdrant's ONNX-based library).

~10× faster than sentence-transformers on CPU; no GPU required.
Models are downloaded once to ~/.cache/fastembed/.

Default model: BAAI/bge-small-en-v1.5  (~22 MB, ~10x faster than jina on CPU)
Code-optimised alternative: jinaai/jina-embeddings-v2-base-code (~311 MB, best quality for code)

Switch model:
    codecompass config set embedding.model jinaai/jina-embeddings-v2-base-code
"""

from __future__ import annotations

from collections.abc import Generator

from codecompass.providers.base import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 256) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name)
        return self._model

    def preload(self) -> None:
        """Explicitly load the ONNX model into memory (and download if needed).

        Call this once before a batch job so the model warm-up cost is visible
        to the user rather than silently freezing the first embed() call.
        """
        self._load()

    def stream_embed(self, texts: list[str]) -> Generator[list[float], None, None]:
        """Yield embeddings one-by-one from a single fastembed generator call.

        Passes all texts in one call so fastembed handles internal batching
        efficiently (no per-batch Python/ONNX overhead). Callers get per-item
        progress by iterating the generator.
        """
        model = self._load()
        for emb in model.embed(texts, batch_size=self._batch_size):
            yield emb.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return list(self.stream_embed(texts))

    @property
    def dimension(self) -> int:
        if self._dim is None:
            sample = list(self._load().embed(["x"]))
            self._dim = len(sample[0])
        return self._dim
