from __future__ import annotations

from codecompass.providers.base import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # lazy-loaded on first embed() call

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        vectors = self._model.encode(
            texts,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
