# code-compass

> Ingest any code repository, build a searchable knowledge base + dependency graph, and answer natural-language questions about the code — with file path and line-range citations.

## What it does

- **Ingest** a local repo or git URL: walks files, chunks with Tree-sitter (AST-aware), embeds chunks, and indexes into ChromaDB + BM25.
- **Retrieve** with hybrid search (dense + BM25, fused with Reciprocal Rank Fusion).
- **Answer** questions via Claude, grounded only in retrieved code with `path:start-end` citations.
- **Evaluate** retrieval quality against a golden question set (Recall@k, MRR).
- **API** — FastAPI `/ask` endpoint.
- **UI** — Streamlit chat interface.
- **MCP server** — Phase 3, for direct integration with Claude Code.

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Add your ANTHROPIC_API_KEY

# 3. Ingest a repo
codecompass ingest /path/to/your/repo
# or from URL:
codecompass ingest https://github.com/psf/requests

# 4. Ask questions
codecompass ask "How does authentication work?"
codecompass ask "Where is the Session class defined?"

# 5. Run the API
codecompass serve

# 6. Open the UI
streamlit run ui/app.py

# 7. Run eval
codecompass eval --golden codecompass/eval/golden/requests_golden.json
```

## Architecture

```
Git repo / URL
     │
     ▼
  INGEST               clone → walk files → AST chunk (Tree-sitter)
     │                 attach metadata: repo, path, language, symbol, lines
     ▼
  INDEX                embed chunks (sentence-transformers)
     │                 → ChromaDB (dense)  +  BM25 index
     ▼
  RETRIEVE             hybrid search: dense + BM25 → RRF fusion → context pack
     │
     ▼
  GENERATE             Claude answers with mandatory citations
     │                 exposed via FastAPI + Streamlit + MCP
     ▼
  EVAL                 Recall@k, MRR, nDCG@k against golden question set
```

## Project structure

```
codecompass/
├── providers/        # LLM, Embedding, VectorStore — swap without touching other code
├── ingest/           # file walking, language detection, Tree-sitter chunking
├── index/            # embed + upsert into ChromaDB; build BM25 index
├── retrieve/         # dense search, BM25 search, hybrid RRF, context packing
├── generate/         # prompts, grounded answering, citation extraction
├── eval/             # golden sets, Recall@k / MRR metrics, eval harness, reports
├── graph/            # dependency graph (NetworkX) from import analysis
├── api/              # FastAPI app
└── cli/              # click CLI: ingest / ask / eval / serve
mcp_server/           # Phase 3: MCP tool server
ui/                   # Streamlit chat UI
tests/                # unit tests (chunker, metrics, context packer, API)
docs/decisions/       # Architecture Decision Records
data/eval_runs/       # timestamped eval results (gitignored)
```

## Configuration

All settings read from `.env` (or environment variables):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for LLM answers |
| `LLM_MODEL_NAME` | `claude-sonnet-4-5` | Claude model to use |
| `EMBEDDING_MODEL_NAME` | `microsoft/codebert-base` | Sentence-transformers model |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector store location |
| `TOP_K_RETRIEVE` | `10` | Chunks retrieved per query |
| `TOKEN_BUDGET` | `6000` | Max tokens for context window |

## Commands

```bash
codecompass ingest <path-or-url>   # build index + graph
codecompass ask "question"          # query with citations
codecompass eval --golden <path>    # run retrieval eval
codecompass serve                   # start FastAPI server
uvicorn codecompass.api.app:app --reload   # dev server
streamlit run ui/app.py             # Streamlit UI
```

## Evaluation

Run against the bundled `requests` library golden set:

```bash
codecompass eval \
  --golden codecompass/eval/golden/requests_golden.json \
  --k 1 --k 5 --k 10
```

Results are printed as a table and saved to `data/eval_runs/eval_<timestamp>.json`.

## Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| LLM | Anthropic Claude (via `anthropic` SDK) |
| Embeddings | `sentence-transformers` (code-capable model) |
| Vector store | ChromaDB (local persistent) |
| Keyword search | `rank-bm25` |
| AST chunking | `tree-sitter` + per-language grammars |
| Dependency graph | NetworkX |
| API | FastAPI + uvicorn |
| UI | Streamlit |
| CLI | click + rich |

## Phases

- **Phase 1 (MVP) ✓** — single-repo RAG with citations, eval harness, API, UI
- **Phase 2** — multi-repo, hybrid retrieval (RRF wired, flip one variable), dependency graph
- **Phase 3** — reranking, query rewriting, MCP server, agentic doc generation
