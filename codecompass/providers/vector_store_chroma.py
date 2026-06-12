from __future__ import annotations

import chromadb

from codecompass.providers.base import SearchResult, VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[SearchResult]:
        kwargs: dict = dict(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        if where is not None:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            SearchResult(
                id=doc_id,
                document=document,
                metadata=metadata,
                # Chroma cosine distance = 1 - similarity, so invert to get similarity score
                score=1.0 - distance,
            )
            for doc_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=False
            )
        ]

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)

    def delete_where(self, where: dict) -> None:
        results = self._collection.get(where=where, include=[])
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
