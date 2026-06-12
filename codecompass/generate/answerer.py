import logging
import re

import tiktoken

from codecompass.config.schema import Config
from codecompass.generate.models import Answer, Citation
from codecompass.generate.prompts import SYSTEM_PROMPT, build_context_block, build_user_message
from codecompass.index.bm25_indexer import BM25Index
from codecompass.providers.base import EmbeddingProvider, LLMProvider, VectorStore
from codecompass.retrieve.context_packer import pack_context
from codecompass.retrieve.hybrid import hybrid_search

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"([\w./\\-]+\.\w+):(\d+)-(\d+)")


def _count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


class Answerer:
    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        store: VectorStore,
        bm25: BM25Index | None,
        config: Config,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._store = store
        self._bm25 = bm25
        self._config = config

    def answer(self, question: str, filters: dict | None = None) -> Answer:
        results = hybrid_search(
            query=question,
            provider=self._embedder,
            store=self._store,
            bm25_index=self._bm25,
            top_k=self._config.retrieval.top_k,
            filters=filters,
        )

        packed = pack_context(results, self._config.retrieval.token_budget, _count_tokens)

        if not packed:
            return Answer(
                text="I don't have enough context to answer this question. No relevant code was retrieved.",
                citations=[],
                question=question,
            )

        context_block = build_context_block(packed)
        user_msg = build_user_message(context_block, question)

        response_text = self._llm.complete(
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=2048,
            system=SYSTEM_PROMPT,
        )

        citations = _extract_citations(response_text, packed)
        return Answer(
            text=response_text,
            citations=citations,
            question=question,
            retrieved_chunk_ids=[r.id for r in packed],
        )

    def stream_answer(self, question: str, filters: dict | None = None):
        """Generator: yields str chunks while streaming, then yields the final Answer."""
        results = hybrid_search(
            query=question,
            provider=self._embedder,
            store=self._store,
            bm25_index=self._bm25,
            top_k=self._config.retrieval.top_k,
            filters=filters,
        )

        packed = pack_context(results, self._config.retrieval.token_budget, _count_tokens)

        if not packed:
            yield Answer(
                text="I don't have enough context to answer this question. No relevant code was retrieved.",
                citations=[],
                question=question,
            )
            return

        context_block = build_context_block(packed)
        user_msg = build_user_message(context_block, question)

        full_text = ""
        for chunk in self._llm.stream_complete(
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=2048,
            system=SYSTEM_PROMPT,
        ):
            full_text += chunk
            yield chunk

        citations = _extract_citations(full_text, packed)
        yield Answer(
            text=full_text,
            citations=citations,
            question=question,
            retrieved_chunk_ids=[r.id for r in packed],
        )


def _extract_citations(text: str, results) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()

    for match in _CITATION_RE.finditer(text):
        path, start, end = match.group(1), int(match.group(2)), int(match.group(3))
        key = f"{path}:{start}-{end}"
        if key in seen:
            continue
        seen.add(key)

        # Find the chunk_id whose line range overlaps this citation
        chunk_id = ""
        for r in results:
            meta = r.metadata
            if meta.get("path") == path:
                r_start = meta.get("start_line", 0)
                r_end = meta.get("end_line", 0)
                if r_start <= start <= r_end or r_start <= end <= r_end:
                    chunk_id = r.id
                    break

        citations.append(Citation(path=path, start_line=start, end_line=end, chunk_id=chunk_id))

    return citations
