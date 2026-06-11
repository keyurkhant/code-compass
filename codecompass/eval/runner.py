from dataclasses import dataclass, field
import json
from pathlib import Path
from codecompass.generate.answerer import Answerer
from codecompass.eval.metrics import recall_at_k, reciprocal_rank, mean_reciprocal_rank, ndcg_at_k


@dataclass
class GoldenEntry:
    question: str
    relevant_paths: list[str]
    reference_answer: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class PerQueryResult:
    question: str
    relevant_paths: list[str]
    retrieved_paths: list[str]
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    rr: float


@dataclass
class EvalResult:
    per_query: list[PerQueryResult]
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    num_queries: int


def load_golden(path: Path) -> list[GoldenEntry]:
    data = json.loads(path.read_text())
    return [GoldenEntry(**entry) for entry in data]


def run_eval(
    golden: list[GoldenEntry],
    answerer: Answerer,
    k_values: list[int] | None = None,
) -> EvalResult:
    if k_values is None:
        k_values = [1, 5, 10]

    per_query: list[PerQueryResult] = []

    for entry in golden:
        # Use the answerer's internal retrieve pipeline
        from codecompass.retrieve.hybrid import hybrid_search
        results = hybrid_search(
            query=entry.question,
            provider=answerer._embedder,
            store=answerer._store,
            bm25_index=answerer._bm25,
            top_k=max(k_values),
        )
        retrieved_paths = [r.metadata.get("path", "") for r in results]

        pq = PerQueryResult(
            question=entry.question,
            relevant_paths=entry.relevant_paths,
            retrieved_paths=retrieved_paths,
            recall_at_1=recall_at_k(retrieved_paths, entry.relevant_paths, 1),
            recall_at_5=recall_at_k(retrieved_paths, entry.relevant_paths, 5),
            recall_at_10=recall_at_k(retrieved_paths, entry.relevant_paths, 10),
            rr=reciprocal_rank(retrieved_paths, entry.relevant_paths),
        )
        per_query.append(pq)

    return EvalResult(
        per_query=per_query,
        recall_at_1=sum(pq.recall_at_1 for pq in per_query) / len(per_query) if per_query else 0.0,
        recall_at_5=sum(pq.recall_at_5 for pq in per_query) / len(per_query) if per_query else 0.0,
        recall_at_10=sum(pq.recall_at_10 for pq in per_query) / len(per_query) if per_query else 0.0,
        mrr=mean_reciprocal_rank([pq.rr for pq in per_query]),
        num_queries=len(per_query),
    )
