# code-compass — Project Blueprint & Claude Code Context

> **One-liner:** code-compass ingests a code repository (and eventually a whole Git org), builds a
> searchable knowledge base plus a dependency graph, and answers natural-language questions about
> the code *with citations*.

This file is written to be used as Claude Code context. Drop it in the repo root (rename to
`CLAUDE.md` if you want Claude Code to load it automatically, or keep this name and `@`-reference it).

---

## 1. Vision & Goals

**Problem.** Understanding an unfamiliar codebase — and how its services/packages depend on each
other across an organization — is slow and mostly tribal knowledge. LLMs can help, but only if they
are given the *right* context. code-compass turns a repo or org into structured, retrievable, citeable
knowledge.

**Two goals held at once:**
1. **Resume value** — demonstrate RAG, embeddings, vector search, graph modeling, agentic tooling,
   evaluation, and clean system design. Be live-demoable.
2. **Real utility** — something you (and others) would actually use to onboard onto code, do impact
   analysis, and generate docs.

**Design principles**
- Ship a complete artifact at every layer. Each phase is presentable on its own.
- Ground every answer in retrieved code; cite file paths and line ranges; say "I don't know" when
  the context doesn't support an answer.
- Measure before optimizing. No quality change ships without an eval number behind it.
- One language path first (Python), then generalize.

---

## 2. Architecture Overview

```
                 +---------------------------------------------+
   Git repo/org  |                  INGESTION                  |
        |        |  clone -> walk files -> structure chunk     |
        v        +---------------+-----------------+-----------+
                                 |                 |
                    +------------v-------+  +-------v-----------+
                    |   KNOWLEDGE BASE   |  | DEPENDENCY GRAPH  |
                    | embeddings + BM25  |  | imports / APIs /  |
                    |  in vector store   |  | service calls     |
                    +------------+-------+  +-------+-----------+
                                 |                  |
                    +------------v------------------v-----------+
                    |              RETRIEVAL LAYER              |
                    | hybrid search -> rerank -> context pack   |
                    +---------------------+---------------------+
                                          |
                    +---------------------v---------------------+
                    |          GENERATION / AGENT LAYER         |
                    |  grounded answers, impact analysis, docs  |
                    |     exposed as API + UI + MCP server      |
                    +-------------------------------------------+
```

---

## 3. Tech Stack (starting choices)

| Concern            | Choice (start)                          | Notes |
|--------------------|-----------------------------------------|-------|
| Language           | Python 3.11+                            | Single ecosystem first. |
| Vector store       | Chroma (local) -> Qdrant                | Swap when you outgrow local. |
| Embeddings         | A current code-capable embedding model  | Evaluate 2-3; don't assume — measure (see §7). |
| Keyword search     | BM25 (`rank_bm25` or built into Qdrant) | Code needs exact-identifier matching. |
| Reranker           | A cross-encoder reranker                | Optional in MVP, big quality win later. |
| Structure parsing  | Tree-sitter                             | Language-agnostic AST chunking. |
| Graph              | NetworkX (start) -> Neo4j               | NetworkX is enough to demo. |
| LLM                | Anthropic / OpenAI API                  | Keep provider behind one interface. |
| API                | FastAPI                                 | Clean, typed, async. |
| UI                 | Streamlit (fast) -> Next.js             | Streamlit to validate, Next.js to polish. |
| Agent/integration  | MCP server                              | The resume differentiator (Phase 3). |
| Eval               | Custom harness + LLM-as-judge           | See §8. |

> Keep every external dependency behind a thin interface (`EmbeddingProvider`, `LLMProvider`,
> `VectorStore`) so you can swap implementations without touching the rest of the code. This is also
> what lets the optimization loop in §9 change one variable at a time.

---

## 4. Proposed Repository Structure

```
code-compass/
├── code-compass/
│   ├── ingest/         # cloning, file walking, chunking
│   ├── index/          # embedding + vector store + bm25
│   ├── graph/          # dependency extraction + graph model
│   ├── retrieve/       # hybrid search, rerank, context packing
│   ├── generate/       # prompts, grounded answering, agents
│   ├── eval/           # eval harness, datasets, metrics
│   ├── providers/      # LLM / embedding / store interfaces
│   ├── api/            # FastAPI app
│   └── config.py
├── mcp_server/         # Phase 3 MCP integration
├── ui/                 # Streamlit or Next.js
├── data/               # local indexes, eval sets (gitignored where large)
├── tests/
├── CLAUDE.md           # (this file, optionally)
└── README.md
```

---

## 5. Implementation Plan — Phased

### Phase 0 — Setup (½ day)
- [ ] Init repo, `pyproject.toml`, virtualenv, pre-commit (ruff + black), pytest.
- [ ] Define provider interfaces in `providers/` (LLM, Embedding, VectorStore) with one concrete impl each.
- [ ] Add `config.py` reading from env (API keys, model names, store path).
- [ ] CI: lint + tests on push.

### Phase 1 — MVP: single-repo RAG with citations (the core deliverable)
**Goal:** Point at one repo, ask questions, get answers grounded in real files with path + line citations.

1. **Ingest** (`ingest/`)
   - [ ] Clone or read a local repo path.
   - [ ] Walk files; respect `.gitignore`; skip binaries, lockfiles, `node_modules`, build dirs.
   - [ ] Detect language by extension.
2. **Chunk** (`ingest/`)
   - [ ] Structure-aware chunking with Tree-sitter: split on function/class boundaries, not blind token windows.
   - [ ] Attach metadata to every chunk: `repo`, `path`, `language`, `symbol_name`, `start_line`, `end_line`.
   - [ ] Prepend lightweight context to each chunk (file path + enclosing class/function signature + relevant imports).
3. **Index** (`index/`)
   - [ ] Embed chunks; upsert into the vector store with metadata.
   - [ ] Build a BM25 index over the same chunks (text + identifiers).
4. **Retrieve** (`retrieve/`)
   - [ ] Dense search top-k.
   - [ ] (MVP can stop at dense; add hybrid in Phase 2 — but design the interface for it now.)
5. **Generate** (`generate/`)
   - [ ] System prompt: answer only from retrieved context, cite `path:start-end`, say "not found" otherwise.
   - [ ] Assemble a token-budget-aware context block from retrieved chunks.
6. **Interface**
   - [ ] FastAPI `/ask` endpoint + a Streamlit chat UI that shows citations as clickable file references.
7. **Eval (do not skip)**
   - [ ] Hand-write 15–25 golden questions about a known repo (e.g. a small popular OSS project), each with the file(s) that *should* be retrieved and a reference answer.
   - [ ] Record baseline recall@k and a faithfulness spot-check. This baseline is the foundation for §9.

**Phase 1 done = a real, demoable project.** Stop here and write it up if you need a deadline.

### Phase 2 — Multi-repo + dependency graph
1. **Multi-repo ingest**
   - [ ] Enumerate an org's repos via the Git host API; ingest each; namespace everything by `repo`.
   - [ ] Add metadata filtering to retrieval (by repo / language / path).
2. **Hybrid retrieval**
   - [ ] Combine dense + BM25 (reciprocal rank fusion). Code questions lean heavily on exact identifiers, so this is a real win.
3. **Dependency graph** (`graph/`)
   - [ ] Static import graph per language (Python `ast`; JS/TS via a dependency tool; multi-lang via Tree-sitter).
   - [ ] Service-level edges where available: parse OpenAPI specs, gRPC stubs, queue topic names, config files.
   - [ ] Store nodes (files/modules/services) and typed edges (`imports`, `provides_api`, `consumes_api`, `depends_on`).
   - [ ] Visualize (NetworkX + a graph view in the UI).
   - ⚠️ **Be honest about limits:** static analysis reliably catches package imports but misses runtime
     dependencies (queues, dynamic dispatch, feature flags). Note this in the README — acknowledging it
     is itself a maturity signal.
4. **Graph-aware answers**
   - [ ] "What depends on X?" / "What breaks if I change Y?" answered by querying the graph, then retrieving code for the affected nodes.

### Phase 3 — Agentic + MCP server (the differentiator)
1. **Reranking** — retrieve top-50, cross-encoder rerank, keep top-8. Measure the lift.
2. **Query rewriting / multi-query** — turn a vague question into 2–3 targeted search queries; fuse results.
3. **Agentic tasks**
   - [ ] Auto-generate onboarding docs / architecture summary for a service.
   - [ ] Cross-repo impact analysis combining graph traversal + retrieval.
4. **MCP server** (`mcp_server/`)
   - [ ] Expose tools: `search_code`, `explain_symbol`, `impact_of_change`, `service_dependencies`.
   - [ ] Now code-compass plugs directly into Claude Code / other MCP clients — strong portfolio moment.

---

## 6. Coding Conventions (for Claude Code)

- Type hints everywhere; functions do one thing; prefer pure functions in `retrieve/` and `ingest/`.
- All external services (LLM, embeddings, store) accessed only through `providers/` interfaces.
- No hidden global state; pass config explicitly.
- Every quality-affecting change must be runnable through the eval harness (§8) before it's considered done.
- Tests: unit-test chunking and graph extraction (deterministic); use a tiny fixture repo in `tests/fixtures/`.
- Commits: small, with a one-line rationale. When changing retrieval/generation, reference the eval delta in the commit body.

**Common commands** (fill in as you build):
```bash
code-compass ingest <repo-path-or-url>     # build index + graph
code-compass ask "how does auth work?"      # query
code-compass eval                           # run the eval harness
uvicorn code-compass.api.app:app --reload   # API
streamlit run ui/app.py                  # UI
```

---

## 7. Choosing Models (don't guess — measure)

Embedding and reranker model quality changes fast, and the "best" model depends on *your* code and
queries. So:
- Shortlist 2–3 current code-capable embedding models.
- Index the same eval corpus with each.
- Compare retrieval metrics (§8) on your golden set.
- Pick the winner; record the decision and the numbers in `docs/decisions/`.

Treat the LLM, embedding model, and reranker as swappable variables, not fixed truths.

---

## 8. Evaluation Harness (`eval/`)

Most portfolio projects skip this. Having it is the single biggest credibility multiplier.

**Build a golden dataset**
- 20–50 questions over a known repo.
- Each entry: `question`, `relevant_file_paths` (what *should* be retrieved), `reference_answer`.
- Cover types: factual ("where is X defined"), flow ("trace request handling"), config, and cross-repo.

**Retrieval metrics** (cheap, deterministic, run on every change)
- `recall@k` — did the relevant files make it into the top-k?
- `MRR` / `nDCG@k` — are they ranked near the top?

**Generation metrics**
- *Faithfulness/groundedness*: is every claim supported by the retrieved context? (LLM-as-judge + human spot-check.)
- *Answer correctness*: vs the reference answer (LLM-as-judge).
- *Citation validity*: do cited paths/lines actually exist and contain the claim?

**Report** — `code-compass eval` prints a table and writes a timestamped JSON to `data/eval_runs/` so
you can chart improvement over time. Put a "Results" section with this table in your README.

---

## 9. Optimization Playbook (the AI-runnable loop)

This is the part an AI agent (Claude Code) can execute autonomously. It has two pieces: **the loop**
(how to improve safely) and **the levers** (what to try).

### 9a. The iterative loop — run this for every improvement
1. **Freeze a baseline.** Ensure the golden set (§8) and a config snapshot are committed.
2. **Measure.** Run `code-compass eval`; record metrics. This is the number to beat.
3. **Form one hypothesis.** e.g. "AST-boundary chunking will raise recall@10 vs fixed-window."
4. **Change exactly one variable.** Never two at once — you won't know which one moved the metric.
5. **Re-measure** on the *same* golden set.
6. **Decide:** keep the change only if the metric improves beyond run-to-run noise (run eval 3×, compare to the noise band). Otherwise revert.
7. **Log the experiment** to `docs/experiments.md`: hypothesis, change, before/after metrics, decision.
8. **Repeat**, tackling the lever with the largest expected impact first.

> Rule for the agent: *no quality change is "done" until step 7 is written.* This makes the whole
> optimization process legible and reproducible — and gives you a great resume write-up for free.

### 9b. The levers — what to try, roughly in order of impact

**Chunking**
- Split on function/class boundaries (Tree-sitter), not fixed token windows.
- Keep chunks self-contained: include the enclosing signature and key imports.
- Tune chunk size; too large dilutes relevance, too small loses context. Sweep a few sizes, measure.

**Retrieval strategy**
- Add **hybrid search** (dense + BM25, fused with reciprocal rank fusion). Usually the biggest single win for code.
- Add **metadata filtering** (repo, language, path) to cut noise before ranking.
- Add a **reranker**: over-retrieve (top-50), cross-encoder rerank, keep top-8.

**Query understanding**
- **Query rewriting**: LLM rephrases the user question into precise search queries.
- **Multi-query / fan-out**: generate several queries, retrieve for each, fuse, dedupe.
- Optionally **HyDE** (embed a hypothetical answer) — measure; it doesn't always help on code.

**Context assembly**
- Token-budget-aware packing: rank, dedupe, and fit the highest-value chunks to the budget.
- Include a compact repo/symbol map so the model knows what exists beyond the retrieved chunks.

**Generation**
- Tighten the system prompt: strict grounding, mandatory citations, explicit "I don't know."
- Use structured output for impact-analysis / doc-generation tasks.
- Prompt-optimize against the eval set the same way as everything else — one change, measure, keep/revert.

**Embeddings / reranker**
- Re-run the model bake-off (§7) once the rest is stable — a better model on top of a good pipeline compounds.

---

## 10. Stretch Ideas (future)
- Incremental re-indexing on git push (only changed files).
- Runtime dependency capture via tracing/service mesh to fill the static-analysis gap.
- Per-team "ownership" overlay on the graph (walk git history for probable owners).
- Natural-language "what changed and why" over a PR or commit range.
- Web demo with a few public repos pre-indexed.

---

## 11. Glossary
- **RAG** — Retrieval-Augmented Generation: retrieve relevant context, then generate grounded answers.
- **Chunk** — a unit of code/text that gets embedded and retrieved.
- **Hybrid search** — combining dense (semantic) and sparse (keyword/BM25) retrieval.
- **Reranker** — a model that re-scores retrieved candidates for relevance (usually a cross-encoder).
- **MCP** — Model Context Protocol: lets the tool be called directly by Claude Code and other clients.
- **Golden set** — curated question/answer/relevant-file dataset used to measure quality.
- **recall@k / MRR / nDCG** — retrieval quality metrics.
