# ADR 001: Embedding Model Choice

**Status:** Proposed
**Date:** 2026-06-11

---

## Context

code-compass needs a dense retrieval embedding model that can understand source code well enough to match natural-language questions to relevant code chunks. Key requirements are:

- Good semantic understanding of code identifiers, function signatures, and docstrings
- Reasonable inference speed on CPU (no GPU assumed in the default deployment)
- No external API dependency for the embedding step (cost and latency concerns)
- Easily swappable via a single environment variable so the team can run comparative evals

---

## Options Considered

| Model | Dims | Notes |
|---|---|---|
| `microsoft/codebert-base` | 768 | Pre-trained on code+NL pairs from GitHub; no API key; runs locally via `sentence-transformers` |
| `jinaai/jina-embeddings-v2-base-code` | 768 | Strong on code; supports 8K context window; also runs locally |
| `openai/text-embedding-3-small` | 1536 | Excellent general quality; requires an OpenAI API key and adds per-token cost |

---

## Decision

Start with **`microsoft/codebert-base`** via the `sentence-transformers` wrapper.

Rationale:
- Requires no API key — the system works fully offline after the model is downloaded.
- CodeBERT was trained on code–natural-language pairs from six programming languages, making it a reasonable baseline for code retrieval.
- The 768-dimensional output fits comfortably in Chroma's HNSW index.
- It is the smallest of the three options, keeping memory footprint low in the default deployment.

---

## Consequences

**Positive**
- Zero API cost and no network dependency at query time.
- Easy to run in CI and offline environments.
- Swapping models requires only changing the `EMBEDDING_MODEL_NAME` environment variable; no code changes.

**Negative / Risks**
- CodeBERT was not fine-tuned for symmetric semantic search; it may underperform on long or ambiguous queries.
- 768 dims may limit recall compared to larger models on nuanced questions.
- The model must be downloaded once (~500 MB); cold-start is slow in ephemeral environments.

**Follow-up**
- Run the `codecompass eval` harness against `requests_golden.json` to establish a Recall@10 / MRR baseline.
- Re-run with `jinaai/jina-embeddings-v2-base-code` and compare; adopt the better model if the gap is meaningful (> 5% MRR).
- Consider `text-embedding-3-small` for hosted deployments where API cost is acceptable.
