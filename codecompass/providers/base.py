from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    id: str
    document: str
    metadata: dict
    score: float


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        system: str | None = None,
    ) -> str: ...

    def stream_complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        system: str | None = None,
    ):
        """Yield text chunks. Default: yield the full complete() result in one piece."""
        yield self.complete(messages, max_tokens=max_tokens, system=system)


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None: ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def delete_collection(self, name: str) -> None: ...

    @abstractmethod
    def delete_where(self, where: dict) -> None:
        """Delete all documents matching the given metadata filter."""
        ...
