"""
MCP server for code-compass.

Exposes code-compass capabilities as MCP tools that Claude Code, Cursor, and
other MCP clients can call via the stdio transport.

To add this server to your MCP client, add the following to its MCP config:

    {
      "mcpServers": {
        "code-compass": {
          "command": "codecompass",
          "args": ["mcp", "serve"]
        }
      }
    }

For Claude Code specifically, run:

    claude mcp add code-compass -- codecompass mcp serve

Or edit ~/.claude/claude_desktop_config.json / .mcp.json directly with the
JSON snippet above.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "code-compass",
    description="RAG-powered code search with dependency graph and cited answers",
)

# ---------------------------------------------------------------------------
# Lazy singletons — nothing heavy is imported/initialised at import time
# ---------------------------------------------------------------------------

_config = None
_answerer = None
_graph = None


def _get_config():
    """Return cached Config, loading from ~/.config/codecompass/config.toml on first call."""
    global _config
    if _config is None:
        from codecompass.config.manager import load_config

        _config = load_config()
    return _config


def _get_answerer():
    """Return cached Answerer, constructing it lazily on first call."""
    global _answerer
    if _answerer is None:
        from codecompass.generate.answerer import Answerer
        from codecompass.index.bm25_indexer import BM25Index
        from codecompass.providers.factory import get_embedding_provider, get_llm_provider
        from codecompass.providers.vector_store_chroma import ChromaVectorStore

        config = _get_config()
        data_dir = config.store.resolved_data_dir()

        llm = get_llm_provider(config)
        embedder = get_embedding_provider(config)

        store = ChromaVectorStore(
            persist_dir=str(data_dir / "chroma"),
            collection_name="codecompass_default",
        )

        bm25 = BM25Index()
        bm25_path = data_dir / "bm25_index.pkl"
        if bm25_path.exists():
            bm25.load(bm25_path)
        else:
            logger.warning(
                "BM25 index not found at %s; falling back to dense-only retrieval.", bm25_path
            )

        # Answerer expects a Settings (pydantic) object for top_k_retrieve / token_budget.
        # We build a minimal shim from the Config dataclass so we don't hard-wire values.
        settings = _config_to_settings(config)

        _answerer = Answerer(llm=llm, embedder=embedder, store=store, bm25=bm25, settings=settings)
    return _answerer


def _get_graph():
    """Return cached DependencyGraph, or None if no graph file exists yet."""
    global _graph
    if _graph is None:
        from codecompass.graph.model import DependencyGraph

        config = _get_config()
        data_dir = config.store.resolved_data_dir()
        graph_path = data_dir / "graph.db"  # stored as JSON despite the .db suffix

        if not graph_path.exists():
            # Try the canonical .json extension as well
            json_path = data_dir / "graph.json"
            if json_path.exists():
                graph_path = json_path
            else:
                return None

        graph = DependencyGraph()
        try:
            graph.load(graph_path)
        except Exception as exc:
            logger.error("Failed to load dependency graph from %s: %s", graph_path, exc)
            return None

        _graph = graph
    return _graph


def _config_to_settings(config):
    """Build a lightweight Settings-compatible object from a Config dataclass.

    Answerer only reads two attributes from its ``settings`` argument:
    ``top_k_retrieve`` and ``token_budget``.  We use a simple namespace so
    we avoid a hard dependency on the pydantic Settings class here.
    """
    import types

    s = types.SimpleNamespace()
    s.top_k_retrieve = config.retrieval.top_k
    s.token_budget = config.retrieval.token_budget
    return s


def _build_filters(language: str = "", repo: str = "") -> dict | None:
    """Translate optional language/repo arguments to a Chroma where-clause."""
    clauses: list[dict] = []
    if language:
        clauses.append({"language": {"$eq": language}})
    if repo:
        clauses.append({"repo": {"$eq": repo}})
    if len(clauses) == 0:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _format_search_results(results) -> str:
    """Render a list of SearchResult objects as a numbered, readable string."""
    if not results:
        return "No results found."

    lines: list[str] = []
    for rank, r in enumerate(results, start=1):
        meta = r.metadata or {}
        path = meta.get("path", "<unknown>")
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")
        score = r.score

        # Grab a brief snippet (first 5 non-empty lines of the document)
        snippet_lines = [ln for ln in (r.document or "").splitlines() if ln.strip()][:5]
        snippet = "\n    ".join(snippet_lines) if snippet_lines else "(no content)"

        lines.append(f"[{rank}] {path}:{start_line}-{end_line} (score: {score:.4f})\n    {snippet}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 1 — search_code
# ---------------------------------------------------------------------------


@mcp.tool()
def search_code(
    query: str,
    top_k: int = 10,
    language: str = "",
    repo: str = "",
) -> str:
    """Search the indexed codebase using hybrid dense + BM25 retrieval.

    Args:
        query:    Natural-language or keyword search query.
        top_k:    Maximum number of results to return (default 10).
        language: Optional language filter, e.g. "python", "go", "javascript".
        repo:     Optional repo name filter to restrict results to one repository.

    Returns:
        Ranked list of code chunks with file path, line range, and a content snippet.
    """
    try:
        from codecompass.retrieve.hybrid import hybrid_search

        answerer = _get_answerer()
        filters = _build_filters(language=language, repo=repo)

        results = hybrid_search(
            query=query,
            provider=answerer._embedder,
            store=answerer._store,
            bm25_index=answerer._bm25,
            top_k=top_k,
            filters=filters,
        )

        return _format_search_results(results)
    except Exception as exc:
        logger.exception("search_code failed")
        return f"Error running search_code: {exc}"


# ---------------------------------------------------------------------------
# Tool 2 — ask_codebase
# ---------------------------------------------------------------------------


@mcp.tool()
def ask_codebase(
    question: str,
    repo: str = "",
) -> str:
    """Ask a natural-language question and get a grounded answer with citations.

    Uses hybrid retrieval to find relevant code, then generates an answer using
    the configured LLM.  The answer includes file:line citations pulled from the
    retrieved context.

    Args:
        question: The question to answer about the codebase.
        repo:     Optional repo name to restrict retrieval to one repository.

    Returns:
        The LLM's answer followed by a formatted list of source citations.
    """
    try:
        answerer = _get_answerer()
        filters = _build_filters(repo=repo)

        answer = answerer.answer(question, filters=filters)

        parts: list[str] = [answer.text]

        if answer.citations:
            parts.append("\n--- Citations ---")
            for i, c in enumerate(answer.citations, start=1):
                parts.append(f"[{i}] {c.path}:{c.start_line}-{c.end_line}")

        return "\n".join(parts)
    except Exception as exc:
        logger.exception("ask_codebase failed")
        return f"Error running ask_codebase: {exc}"


# ---------------------------------------------------------------------------
# Tool 3 — explain_symbol
# ---------------------------------------------------------------------------


@mcp.tool()
def explain_symbol(
    symbol_name: str,
    repo: str = "",
) -> str:
    """Find and explain a specific function, class, or module.

    Searches both the dependency graph (for symbol metadata) and the vector
    index (for the actual source code) then returns the code and a brief
    explanation generated by the LLM.

    Args:
        symbol_name: Name of the function, class, or module to explain.
        repo:        Optional repo name to restrict the search.

    Returns:
        The symbol's source code and an LLM-generated explanation with
        file:line citations.
    """
    try:
        from codecompass.retrieve.hybrid import hybrid_search

        answerer = _get_answerer()
        filters = _build_filters(repo=repo)

        # --- Vector search for the symbol ---
        results = hybrid_search(
            query=f"definition of {symbol_name}",
            provider=answerer._embedder,
            store=answerer._store,
            bm25_index=answerer._bm25,
            top_k=5,
            filters=filters,
        )

        if not results:
            return f"No code found matching symbol '{symbol_name}'."

        # Build a context string from the top results
        context_parts: list[str] = []
        for r in results:
            meta = r.metadata or {}
            path = meta.get("path", "<unknown>")
            start = meta.get("start_line", "?")
            end = meta.get("end_line", "?")
            context_parts.append(f"# {path}:{start}-{end}\n{r.document or ''}")

        context_block = "\n\n".join(context_parts)

        # --- Graph metadata (best-effort) ---
        graph_note = ""
        graph = _get_graph()
        if graph is not None:
            matching = [nid for nid in graph._g.nodes if symbol_name.lower() in nid.lower()]
            if matching:
                node_ids_str = "\n".join(f"  - {nid}" for nid in matching[:10])
                graph_note = f"\nGraph nodes matching '{symbol_name}':\n{node_ids_str}\n"

        # --- Ask the LLM to explain ---
        prompt = (
            f"You are a code expert. Explain the symbol '{symbol_name}' "
            f"based on the following source code.\n\n"
            f"{graph_note}"
            f"Source code:\n```\n{context_block}\n```\n\n"
            f"Provide:\n"
            f"1. A one-sentence summary of what '{symbol_name}' does.\n"
            f"2. Key parameters / attributes (if any).\n"
            f"3. Notable callers or dependencies visible in the context.\n"
            f"Always cite file and line numbers, e.g. path/to/file.py:10-20."
        )

        explanation = answerer._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )

        header = _format_search_results(results[:3])
        return f"Symbol matches:\n{header}\n\n--- Explanation ---\n{explanation}"
    except Exception as exc:
        logger.exception("explain_symbol failed")
        return f"Error running explain_symbol: {exc}"


# ---------------------------------------------------------------------------
# Tool 4 — impact_of_change
# ---------------------------------------------------------------------------


@mcp.tool()
def impact_of_change(
    file_path: str,
    repo: str = "",
    depth: int = 3,
) -> str:
    """Return what else in the codebase would be affected if a file changed.

    Performs a reverse-dependency traversal in the dependency graph starting
    from the given file, up to ``depth`` hops.

    Args:
        file_path: Relative path of the file inside the repository.
        repo:      Repo name (required when multiple repos are indexed).
        depth:     How many hops of reverse dependencies to follow (default 3).

    Returns:
        A formatted list of files/symbols that transitively depend on the file.
    """
    try:
        graph = _get_graph()
        if graph is None:
            return (
                "Dependency graph is not available. "
                "Run 'codecompass ingest' to build the graph first."
            )

        # Node IDs are stored as "repo:path"
        # Try to find the node with a best-effort match
        node_id = _find_node(graph, file_path, repo)
        if node_id is None:
            available = _sample_nodes(graph, 5)
            return (
                f"No graph node found for '{file_path}' "
                f"(repo={repo!r}).\n"
                f"Sample node IDs in the graph:\n{available}"
            )

        affected = graph.impact_of_change(node_id, depth=depth)

        if not affected:
            return f"No dependents found for '{file_path}' within depth {depth}."

        lines = [f"Files/symbols affected by changes to '{file_path}' (depth={depth}):"]
        for nid in affected:
            node_data = graph._g.nodes.get(nid, {})
            node_path = node_data.get("path", nid)
            node_type = node_data.get("node_type", "file")
            lines.append(f"  [{node_type}] {node_path}")

        return "\n".join(lines)
    except Exception as exc:
        logger.exception("impact_of_change failed")
        return f"Error running impact_of_change: {exc}"


# ---------------------------------------------------------------------------
# Tool 5 — get_dependencies
# ---------------------------------------------------------------------------


@mcp.tool()
def get_dependencies(
    file_path: str,
    repo: str = "",
) -> str:
    """Show what a file imports / depends on.

    Looks up the file in the dependency graph and returns its direct and
    transitive dependencies (up to depth 2).

    Args:
        file_path: Relative path of the file inside the repository.
        repo:      Repo name (required when multiple repos are indexed).

    Returns:
        List of files/modules this file depends on, annotated with edge types.
    """
    try:
        graph = _get_graph()
        if graph is None:
            return (
                "Dependency graph is not available. "
                "Run 'codecompass ingest' to build the graph first."
            )

        node_id = _find_node(graph, file_path, repo)
        if node_id is None:
            available = _sample_nodes(graph, 5)
            return (
                f"No graph node found for '{file_path}' "
                f"(repo={repo!r}).\n"
                f"Sample node IDs in the graph:\n{available}"
            )

        lines = [f"Dependencies of '{file_path}':"]

        # Direct dependencies (depth 1)
        direct = graph.dependencies_of(node_id)
        if not direct:
            lines.append("  (no direct dependencies found in graph)")
        else:
            lines.append("  Direct dependencies:")
            for dep_id in direct:
                edge_data = graph._g.get_edge_data(node_id, dep_id) or {}
                edge_type = edge_data.get("edge_type", "depends_on")
                dep_data = graph._g.nodes.get(dep_id, {})
                dep_path = dep_data.get("path", dep_id)
                lines.append(f"    --[{edge_type}]--> {dep_path}")

            # Depth-2 (transitive)
            transitive: dict[str, str] = {}
            for dep_id in direct:
                for tdep_id in graph.dependencies_of(dep_id):
                    if tdep_id != node_id and tdep_id not in direct:
                        edge_data = graph._g.get_edge_data(dep_id, tdep_id) or {}
                        transitive[tdep_id] = edge_data.get("edge_type", "depends_on")

            if transitive:
                lines.append("  Transitive dependencies (depth 2):")
                for tdep_id, etype in transitive.items():
                    tdep_data = graph._g.nodes.get(tdep_id, {})
                    tdep_path = tdep_data.get("path", tdep_id)
                    lines.append(f"    --[{etype}]--> {tdep_path}")

        return "\n".join(lines)
    except Exception as exc:
        logger.exception("get_dependencies failed")
        return f"Error running get_dependencies: {exc}"


# ---------------------------------------------------------------------------
# Tool 6 — list_repos
# ---------------------------------------------------------------------------


@mcp.tool()
def list_repos() -> str:
    """List all indexed repositories and their graph statistics.

    Returns:
        Formatted list of repos with node and edge counts derived from the
        dependency graph.
    """
    try:
        graph = _get_graph()
        if graph is None:
            return (
                "Dependency graph is not available. "
                "Run 'codecompass ingest' to build the graph first."
            )

        # Collect unique repos from node attributes
        repo_stats: dict[str, dict] = {}
        for nid, data in graph._g.nodes(data=True):
            r = data.get("repo", "")
            if not r:
                # Node ID format is "repo:path"
                r = nid.split(":")[0] if ":" in nid else "<unknown>"
            if r not in repo_stats:
                repo_stats[r] = {"nodes": 0, "edges": 0}
            repo_stats[r]["nodes"] += 1

        # Count edges per repo
        for src, _dst, _data in graph._g.edges(data=True):
            src_repo = graph._g.nodes[src].get("repo", src.split(":")[0] if ":" in src else "")
            if src_repo in repo_stats:
                repo_stats[src_repo]["edges"] += 1

        if not repo_stats:
            return "No repositories found in the dependency graph."

        lines = [f"Indexed repositories ({len(repo_stats)} total):"]
        for repo_name, stats in sorted(repo_stats.items()):
            lines.append(f"  {repo_name}  —  {stats['nodes']} nodes, {stats['edges']} edges")

        return "\n".join(lines)
    except Exception as exc:
        logger.exception("list_repos failed")
        return f"Error running list_repos: {exc}"


# ---------------------------------------------------------------------------
# Tool 7 — graph_stats
# ---------------------------------------------------------------------------


@mcp.tool()
def graph_stats(repo: str = "") -> str:
    """Show dependency graph statistics: total nodes, edges, and type breakdown.

    Args:
        repo: Optional repo name to filter stats to a single repository.

    Returns:
        Text summary of node/edge counts and node-type distribution.
    """
    try:
        graph = _get_graph()
        if graph is None:
            return (
                "Dependency graph is not available. "
                "Run 'codecompass ingest' to build the graph first."
            )

        # Filter to the requested repo (or use all nodes)
        if repo:
            nodes = [
                (nid, data)
                for nid, data in graph._g.nodes(data=True)
                if data.get("repo", nid.split(":")[0] if ":" in nid else "") == repo
            ]
            node_ids = {nid for nid, _ in nodes}
            edges = [
                (s, d, data)
                for s, d, data in graph._g.edges(data=True)
                if s in node_ids and d in node_ids
            ]
        else:
            nodes = list(graph._g.nodes(data=True))
            edges = list(graph._g.edges(data=True))

        total_nodes = len(nodes)
        total_edges = len(edges)

        # Node type breakdown
        type_counts: dict[str, int] = {}
        for _nid, data in nodes:
            nt = data.get("node_type", "unknown")
            type_counts[nt] = type_counts.get(nt, 0) + 1

        # Edge type breakdown
        edge_type_counts: dict[str, int] = {}
        for _s, _d, data in edges:
            et = data.get("edge_type", "unknown")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

        lines = [
            f"Graph statistics{f' for repo {repo!r}' if repo else ''}:",
            f"  Total nodes : {total_nodes}",
            f"  Total edges : {total_edges}",
            "",
            "  Node types:",
        ]
        for nt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {nt}: {count}")

        if edge_type_counts:
            lines.append("  Edge types:")
            for et, count in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {et}: {count}")

        return "\n".join(lines)
    except Exception as exc:
        logger.exception("graph_stats failed")
        return f"Error running graph_stats: {exc}"


# ---------------------------------------------------------------------------
# Private graph helpers
# ---------------------------------------------------------------------------


def _find_node(graph, file_path: str, repo: str) -> str | None:
    """Return the graph node ID for a given file_path and optional repo."""
    # Exact match first: "repo:path"
    if repo:
        candidate = f"{repo}:{file_path}"
        if candidate in graph._g:
            return candidate

    # Fallback: look for any node whose 'path' attribute matches
    for nid, data in graph._g.nodes(data=True):
        node_path = data.get("path", "")
        node_repo = data.get("repo", "")
        if (node_path == file_path or node_path.endswith(f"/{file_path}")) and (
            not repo or node_repo == repo
        ):
            return nid

    # Last resort: substring match on node ID
    for nid in graph._g.nodes:
        if file_path in nid and (not repo or nid.startswith(f"{repo}:")):
            return nid

    return None


def _sample_nodes(graph, n: int) -> str:
    """Return a formatted sample of node IDs for helpful error messages."""
    sample = list(graph._g.nodes)[:n]
    return "\n".join(f"  {nid}" for nid in sample) or "  (graph is empty)"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Start the MCP server using the stdio transport.

    Add to your MCP client config:

        {
          "mcpServers": {
            "code-compass": {
              "command": "codecompass",
              "args": ["mcp", "serve"]
            }
          }
        }
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
