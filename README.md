# code-compass

> RAG-powered code intelligence: ingest any repo, ask natural-language questions, get cited answers and dependency graphs — no API keys required.

Works out of the box with **Claude Code** (free, no key) or **Ollama** (local). Cloud providers (Anthropic, OpenAI, Gemini) are supported via LiteLLM for users who prefer them.

---

## Quickstart (30 seconds)

```bash
# 1. Install with uv
uv pip install -e .

# 2. Point at a provider (Claude Code requires no key)
codecompass config set llm.provider claude-code

# 3. Ingest a repo
codecompass ingest /path/to/your/repo

# 4. Ask questions
codecompass ask "How does authentication work?"
codecompass ask "Where is the Session class defined?"

# 5. Start the API
codecompass serve
```

---

## Provider setup

### Claude Code (no API key)

Requires the `claude` CLI to be installed and authenticated.

```bash
codecompass config set llm.provider claude-code
```

Auto-detection: if the `claude` binary is on your `PATH`, code-compass selects it automatically (no config needed).

### Ollama (local, no API key)

```bash
# Pull a model first
ollama pull llama3.2

# Configure
codecompass config set llm.provider ollama
codecompass config set llm.model llama3.2
# Optional — only needed if Ollama is not on localhost:11434
codecompass config set llm.base_url http://localhost:11434
```

### Cloud provider (API key required)

```bash
# Anthropic Claude
uv pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
codecompass config set llm.provider litellm
codecompass config set llm.model claude-sonnet-4-5

# OpenAI
uv pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
codecompass config set llm.provider litellm
codecompass config set llm.model gpt-4o

# Google Gemini
uv pip install -e ".[google]"
export GEMINI_API_KEY=...
codecompass config set llm.provider litellm
codecompass config set llm.model gemini/gemini-1.5-pro

# Any OpenAI-compatible endpoint (LM Studio, vLLM, etc.)
codecompass config set llm.provider openai-compat
codecompass config set llm.base_url http://localhost:1234/v1
codecompass config set llm.model local-model
```

---

## Commands

### `codecompass config`

Read and write settings stored in `~/.config/codecompass/config.toml`. Environment variables override the file.

```bash
codecompass config set llm.provider claude-code
codecompass config set llm.model llama3.2
codecompass config set llm.base_url http://localhost:11434
codecompass config set embedding.model BAAI/bge-small-en-v1.5
codecompass config set retrieval.top_k 15
codecompass config set store.data_dir /mnt/data/codecompass

codecompass config get llm.provider
codecompass config list
codecompass config unset llm.model
```

Config keys:

| Key | Default | Description |
|-----|---------|-------------|
| `llm.provider` | `auto` | `auto`, `claude-code`, `ollama`, `litellm`, `openai-compat`, `subprocess` |
| `llm.model` | *(provider default)* | Model name passed to the provider |
| `llm.base_url` | *(provider default)* | Override endpoint URL |
| `llm.api_key` | *(env var)* | API key (prefer env vars) |
| `llm.timeout` | `120` | Request timeout in seconds |
| `embedding.model` | `jinaai/jina-embeddings-v2-base-code` | FastEmbed model |
| `embedding.batch_size` | `32` | Embedding batch size |
| `store.data_dir` | `~/.codecompass` | Where indexes and graph DB are stored |
| `retrieval.top_k` | `10` | Chunks retrieved per query |
| `retrieval.token_budget` | `6000` | Max context tokens sent to the LLM |

### `codecompass ingest`

Walk a repo, chunk with Tree-sitter (AST-aware), embed, index into ChromaDB + BM25, and extract a dependency graph.

```bash
codecompass ingest /path/to/repo
codecompass ingest https://github.com/psf/requests --name requests
codecompass ingest /path/to/repo --force   # rebuild existing index
```

Data is written to `store.data_dir` (default `~/.codecompass`):
- `chroma/` — vector store
- `bm25_index.pkl` — BM25 index
- `graph.db` — SQLite dependency graph

### `codecompass ask`

Ask a natural-language question. Returns a cited answer with `file:start-end` references.

```bash
codecompass ask "how does auth work?"
codecompass ask "where is Session defined?" --repo-name requests
codecompass ask "explain chunking" --repo /path/to/repo   # auto-ingest first
codecompass ask "list all route handlers" --filter-lang python
codecompass ask "find cache logic" --filter-path cache --top-k 20
```

### `codecompass graph`

Inspect the SQLite-backed dependency graph extracted during ingest.

```bash
# Overview stats + top symbols for a repo
codecompass graph show myrepo

# What breaks if I change this file?
codecompass graph impact codecompass/retrieve/hybrid.py --repo code-compass

# What does this file depend on?
codecompass graph deps codecompass/generate/answerer.py --repo code-compass --depth 2
```

### `codecompass eval`

Run the retrieval evaluation harness against a golden question set (Recall@k, MRR).

```bash
codecompass eval --golden codecompass/eval/golden/requests_golden.json
codecompass eval --golden golden.json --k 1 --k 5 --k 10 --repo-name requests
```

Results are printed as a table and saved to `./data/eval_results/eval_<timestamp>.json`.

### `codecompass serve`

Start the FastAPI server (`GET /health`, `POST /ask`).

```bash
codecompass serve
codecompass serve --port 8080
codecompass serve --reload   # dev mode with auto-reload
```

### `codecompass run`

Ingest a repo and start the server in one command.

```bash
codecompass run /path/to/repo
codecompass run https://github.com/psf/requests --port 8080
```

### MCP server

code-compass exposes an MCP server so AI coding tools can call it as a tool.

```bash
codecompass mcp serve   # start server (stdio transport)
```

**Add to Claude Code** (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "code-compass": {
      "command": "codecompass",
      "args": ["mcp", "serve"]
    }
  }
}
```

**Add to Cursor** (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "code-compass": {
      "command": "codecompass",
      "args": ["mcp", "serve"]
    }
  }
}
```

---

## Architecture

```
Git repo / URL
      |
      v
   INGEST          clone -> walk files -> AST chunk (Tree-sitter)
      |             attach metadata: repo, path, language, symbol, lines
      v
   INDEX           embed chunks (FastEmbed / ONNX, no GPU needed)
      |             -> ChromaDB (dense)  +  BM25 index  +  SQLite graph
      v
   RETRIEVE        hybrid search: dense + BM25 -> RRF fusion -> context pack
      |
      v
   GENERATE        LLM answers with mandatory path:start-end citations
      |             provider: Claude Code | Ollama | LiteLLM | subprocess
      v
   EVAL            Recall@k, MRR against golden question set
      |
      +---> FastAPI /ask endpoint
      +---> MCP server (stdio transport)
```

### Project structure

```
codecompass/
  providers/      LLM, Embedding, VectorStore providers + factory
  config/         Config dataclass, TOML manager, schema
  ingest/         file walking, language detection, Tree-sitter chunking
  index/          embed + upsert into ChromaDB; build BM25 index
  retrieve/       dense search, BM25 search, hybrid RRF, context packing
  generate/       prompts, grounded answering, citation extraction
  eval/           golden sets, Recall@k / MRR metrics, eval harness
  graph/          SQLite dependency graph from import/call analysis
  api/            FastAPI app (routers: /health, /ask)
  cli/            click CLI: config / ingest / ask / serve / graph / mcp
mcp_server/       MCP server (stdio transport)
tests/            pytest unit tests
```

---

## Tech stack

| Concern | Choice |
|---------|--------|
| Language | Python 3.11+ |
| Package manager | uv |
| LLM | Claude Code / Ollama / LiteLLM (any provider) |
| Embeddings | FastEmbed (ONNX, ~10x faster than sentence-transformers, no GPU) |
| Vector store | ChromaDB (local persistent) |
| Keyword search | rank-bm25 |
| AST chunking | tree-sitter + per-language grammars |
| Dependency graph | SQLite (recursive CTEs for BFS/DFS, FTS5 search) |
| API | FastAPI + uvicorn |
| CLI | click + rich |
| MCP | mcp (stdio transport) |

---

## Evaluation

Run against the bundled `requests` library golden set:

```bash
codecompass eval \
  --golden codecompass/eval/golden/requests_golden.json \
  --k 1 --k 5 --k 10
```

Metrics reported: Recall@1, Recall@5, Recall@10, MRR.

Results are saved to `data/eval_results/eval_<timestamp>.json`.

---

## Development

```bash
# Install with dev extras
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint + format
ruff check .
ruff format .

# Install pre-commit hooks
pre-commit install
```
