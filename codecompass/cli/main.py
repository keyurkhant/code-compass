"""
code-compass CLI.

Commands:
  config  -- Read and write settings in ~/.config/codecompass/config.toml.
  ingest  -- Walk a repo, chunk, embed, build graph, and index source files.
  ask     -- Ask a question. Pass --repo to auto-ingest first.
  run     -- Ingest a repo then start the API server (one command, no fuss).
  eval    -- Run the retrieval evaluation harness against a golden set.
  serve   -- Start the FastAPI server.
  graph   -- Inspect the dependency graph (stats, impact, deps).
  mcp     -- MCP server commands.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from codecompass.cli.config_cmd import config_group

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_providers(config):
    """Instantiate LLM + embedding providers from config."""
    from codecompass.providers.factory import get_embedding_provider, get_llm_provider

    embedder = get_embedding_provider(config)
    llm = get_llm_provider(config)
    return embedder, llm


def _build_store(config, collection_name: str):
    from codecompass.providers.vector_store_chroma import ChromaVectorStore

    data_dir = config.store.resolved_data_dir()
    return ChromaVectorStore(
        persist_dir=str(data_dir / "chroma"),
        collection_name=collection_name,
    )


def _do_ingest(repo_path_or_url: str, repo_name: str | None, force: bool, config) -> str:
    """Core ingest logic. Returns the collection name used."""
    from codecompass.graph.extractor import build_graph
    from codecompass.graph.model import GraphDB
    from codecompass.index.bm25_indexer import BM25Index
    from codecompass.index.vector_indexer import index_chunks
    from codecompass.ingest.reader import ingest_repo

    embedder, _ = _build_providers(config)

    data_dir = config.store.resolved_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = data_dir / "bm25_index.pkl"
    graph_db_path = data_dir / "graph.db"

    # Step 1: walk + chunk
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Walking repository and chunking files...", total=None)
        try:
            chunks, repo_root = ingest_repo(repo_path_or_url, repo_name=repo_name)
        except Exception as exc:
            console.print(f"[red]Ingest failed:[/red] {exc}")
            sys.exit(1)
        p.update(task, description=f"Chunked {len(chunks)} code chunks", completed=1, total=1)

    derived_name = repo_name or (chunks[0].repo if chunks else "default")
    collection_name = f"codecompass_{derived_name}"
    console.print(f"Collection: [cyan]{collection_name}[/cyan]")

    store = _build_store(config, collection_name)
    if force:
        console.print("[yellow]--force:[/yellow] rebuilding existing collection...")
        try:
            store.delete_collection(collection_name)
            store = _build_store(config, collection_name)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] {exc}")

    # Step 2a: load the embedding model (download on first use, ~300 MB for default model)
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task(
            f"Loading embedding model [cyan]{config.embedding.model}[/cyan]"
            " (first run downloads the model, may take a minute)...",
            total=None,
        )
        embedder.preload()
        p.update(task, description="Embedding model ready", completed=1, total=1)

    # Step 2b: embed
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as p:
        task = p.add_task("Embedding chunks...", total=len(chunks))
        batch_size = config.embedding.batch_size
        chunks_with_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.embed_text() for c in batch]
            embeddings = embedder.embed(texts)
            chunks_with_embeddings.extend(zip(batch, embeddings, strict=False))
            p.advance(task, len(batch))

    # Step 3: vector store
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Indexing into vector store...", total=None)
        count = index_chunks(chunks_with_embeddings, store)
        p.update(task, description=f"Indexed {count} chunks", completed=1, total=1)

    # Step 4: BM25
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Building BM25 index...", total=None)
        bm25 = BM25Index()
        bm25.build(chunks)
        bm25.save(bm25_path)
        p.update(task, description=f"BM25 saved to {bm25_path}", completed=1, total=1)

    # Step 5: graph extraction
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Building dependency graph...", total=None)
        graph_db = GraphDB(graph_db_path)
        try:
            graph_stats = build_graph(
                repo_root=Path(repo_root) if not isinstance(repo_root, Path) else repo_root,
                repo=derived_name,
                graph=graph_db,
                incremental=True,
            )
        finally:
            graph_db.close()
        p.update(
            task,
            description=(
                f"Graph: {graph_stats['files_processed']} files processed, "
                f"{graph_stats['nodes_added']} nodes added"
            ),
            completed=1,
            total=1,
        )

    console.print(
        Panel(
            f"[green]Done![/green] {len(chunks)} chunks -> collection [cyan]{collection_name}[/cyan]",
            title="Ingest complete",
        )
    )
    return collection_name


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """code-compass: RAG-powered code search and Q&A with citations."""


# Register subgroups
cli.add_command(config_group)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@cli.command("ingest")
@click.argument("repo_path_or_url")
@click.option(
    "--name", "repo_name", default=None, help="Override the repo name / collection prefix."
)
@click.option(
    "--force/--no-force", default=False, help="Delete and rebuild the existing collection."
)
def ingest_cmd(repo_path_or_url: str, repo_name: str | None, force: bool) -> None:
    """Ingest a local repo or remote git URL into the vector index.

    \b
        codecompass ingest /path/to/repo
        codecompass ingest https://github.com/psf/requests --name requests
    """
    from codecompass.config.manager import load_config

    _do_ingest(repo_path_or_url, repo_name, force, load_config())


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


@cli.command("ask")
@click.argument("question")
@click.option(
    "--repo", "repo_path_or_url", default=None, help="Auto-ingest this repo first if not indexed."
)
@click.option("--repo-name", default=None, help="Repo name / collection to query.")
@click.option("--top-k", default=None, type=int, help="Chunks to retrieve (overrides config).")
@click.option("--filter-lang", default=None, help="Restrict to a language (python, go, ...).")
@click.option("--filter-path", default=None, help="Restrict to paths containing this substring.")
def ask_cmd(
    question: str,
    repo_path_or_url: str | None,
    repo_name: str | None,
    top_k: int | None,
    filter_lang: str | None,
    filter_path: str | None,
) -> None:
    """Ask a question about the indexed codebase.

    Pass --repo to auto-ingest a repository before answering:

    \b
        codecompass ask "how does auth work?" --repo /path/to/repo
        codecompass ask "where is Session defined?" --repo https://github.com/psf/requests
        codecompass ask "explain the chunking logic" --repo-name myrepo
    """
    from codecompass.config.manager import load_config
    from codecompass.generate.answerer import Answerer
    from codecompass.index.bm25_indexer import BM25Index

    config = load_config()

    # Apply --top-k override
    if top_k is not None:
        config.retrieval.top_k = top_k

    data_dir = config.store.resolved_data_dir()
    bm25_path = data_dir / "bm25_index.pkl"

    # Auto-ingest if the caller provided a repo path
    if repo_path_or_url:
        collection_name = _do_ingest(repo_path_or_url, repo_name, force=False, config=config)
    else:
        derived = repo_name or "default"
        collection_name = f"codecompass_{derived}"

    embedder, llm = _build_providers(config)
    store = _build_store(config, collection_name)

    bm25 = BM25Index()
    if bm25_path.exists():
        bm25.load(bm25_path)
    else:
        console.print("[yellow]Warning:[/yellow] BM25 index not found; using dense-only retrieval.")

    # Build Chroma where-clause filters
    where_clauses: list[dict] = []
    if filter_lang:
        where_clauses.append({"language": {"$eq": filter_lang}})
    if filter_path:
        where_clauses.append({"path": {"$contains": filter_path}})
    filters: dict | None = None
    if len(where_clauses) == 1:
        filters = where_clauses[0]
    elif len(where_clauses) > 1:
        filters = {"$and": where_clauses}

    answerer = Answerer(llm=llm, embedder=embedder, store=store, bm25=bm25, config=config)

    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")

    with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
        try:
            answer = answerer.answer(question, filters=filters)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

    console.print(Panel(answer.text, title="Answer", border_style="green"))

    if answer.citations:
        table = Table(title="Citations", show_header=True, header_style="bold magenta")
        table.add_column("File", style="cyan")
        table.add_column("Lines", justify="right")
        for c in answer.citations:
            table.add_row(c.path, f"{c.start_line}-{c.end_line}")
        console.print(table)


# ---------------------------------------------------------------------------
# run  (ingest + serve in one shot)
# ---------------------------------------------------------------------------


@cli.command("run")
@click.argument("repo_path_or_url")
@click.option("--name", "repo_name", default=None, help="Override the repo name.")
@click.option("--force/--no-force", default=False, help="Rebuild index before serving.")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
def run_cmd(
    repo_path_or_url: str,
    repo_name: str | None,
    force: bool,
    host: str,
    port: int,
) -> None:
    """Ingest a repo then start the API server -- one command, no fuss.

    \b
        codecompass run /path/to/repo
        codecompass run https://github.com/psf/requests --port 8080
    """
    from codecompass.config.manager import load_config

    config = load_config()
    _do_ingest(repo_path_or_url, repo_name, force, config)

    console.print(
        f"\n[bold green]Starting server[/bold green] on [cyan]http://{host}:{port}[/cyan]\n"
    )

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "codecompass.api.app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Server error:[/red] {exc}")
        sys.exit(exc.returncode)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@cli.command("eval")
@click.option(
    "--golden",
    "golden_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the golden JSON file.",
)
@click.option("--k", "k_values", multiple=True, type=int, default=[1, 5, 10], show_default=True)
@click.option(
    "--output-dir", "output_dir", default="./data/eval_results", type=click.Path(path_type=Path)
)
@click.option("--repo-name", default=None, help="Target repo collection (default: default).")
def eval_cmd(
    golden_path: Path,
    k_values: tuple[int, ...],
    output_dir: Path,
    repo_name: str | None,
) -> None:
    """Run the retrieval evaluation harness against a golden question set.

    \b
        codecompass eval --golden codecompass/eval/golden/requests_golden.json
        codecompass eval --golden golden.json --k 1 --k 5 --k 10 --repo-name myrepo
    """
    from codecompass.config.manager import load_config
    from codecompass.eval import metrics as _m
    from codecompass.eval.report import print_eval_table, save_eval_result
    from codecompass.eval.runner import EvalResult, PerQueryResult, load_golden
    from codecompass.generate.answerer import Answerer
    from codecompass.index.bm25_indexer import BM25Index
    from codecompass.retrieve.hybrid import hybrid_search

    config = load_config()
    embedder, llm = _build_providers(config)
    derived = repo_name or "default"
    collection_name = f"codecompass_{derived}"
    store = _build_store(config, collection_name)

    data_dir = config.store.resolved_data_dir()
    bm25 = BM25Index()
    bm25_path = data_dir / "bm25_index.pkl"
    if bm25_path.exists():
        bm25.load(bm25_path)

    answerer = Answerer(llm=llm, embedder=embedder, store=store, bm25=bm25, config=config)

    console.print(f"Loading golden set from [cyan]{golden_path}[/cyan]...")
    golden = load_golden(golden_path)
    console.print(f"Loaded [bold]{len(golden)}[/bold] questions.")

    k_list = list(k_values)
    per_query: list[PerQueryResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as p:
        task = p.add_task("Running eval queries...", total=len(golden))
        for entry in golden:
            results = hybrid_search(
                query=entry.question,
                provider=answerer._embedder,
                store=answerer._store,
                bm25_index=answerer._bm25,
                top_k=max(k_list),
            )
            retrieved_paths = [r.metadata.get("path", "") for r in results]
            per_query.append(
                PerQueryResult(
                    question=entry.question,
                    relevant_paths=entry.relevant_paths,
                    retrieved_paths=retrieved_paths,
                    recall_at_1=_m.recall_at_k(retrieved_paths, entry.relevant_paths, 1),
                    recall_at_5=_m.recall_at_k(retrieved_paths, entry.relevant_paths, 5),
                    recall_at_10=_m.recall_at_k(retrieved_paths, entry.relevant_paths, 10),
                    rr=_m.reciprocal_rank(retrieved_paths, entry.relevant_paths),
                )
            )
            p.advance(task)

    result = EvalResult(
        per_query=per_query,
        recall_at_1=sum(pq.recall_at_1 for pq in per_query) / len(per_query) if per_query else 0.0,
        recall_at_5=sum(pq.recall_at_5 for pq in per_query) / len(per_query) if per_query else 0.0,
        recall_at_10=sum(pq.recall_at_10 for pq in per_query) / len(per_query)
        if per_query
        else 0.0,
        mrr=_m.mean_reciprocal_rank([pq.rr for pq in per_query]),
        num_queries=len(per_query),
    )

    print_eval_table(result)
    saved_path = save_eval_result(result, output_dir)
    console.print(f"[green]Results saved to[/green] [cyan]{saved_path}[/cyan]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (dev mode).")
def serve_cmd(host: str, port: int, reload: bool) -> None:
    """Start the code-compass FastAPI server.

    \b
        codecompass serve
        codecompass serve --port 8080
        codecompass serve --reload   # dev mode
    """
    console.print(f"[bold green]Starting server[/bold green] on [cyan]http://{host}:{port}[/cyan]")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "codecompass.api.app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Server error:[/red] {exc}")
        sys.exit(exc.returncode)


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


@click.group("graph")
def graph_group() -> None:
    """Inspect the code dependency graph."""


cli.add_command(graph_group)


@graph_group.command("show")
@click.argument("repo_name")
def graph_show(repo_name: str) -> None:
    """Show graph statistics and top nodes for a repo.

    \b
        codecompass graph show myrepo
    """
    from collections import Counter

    from codecompass.config.manager import load_config
    from codecompass.graph.model import GraphDB

    config = load_config()
    graph_db_path = config.store.resolved_data_dir() / "graph.db"

    if not graph_db_path.exists():
        console.print(f"[red]Graph database not found at {graph_db_path}.[/red]")
        console.print("Run [cyan]codecompass ingest <repo>[/cyan] first.")
        sys.exit(1)

    graph_db = GraphDB(graph_db_path)
    try:
        stats = graph_db.stats()
        nodes = graph_db.nodes_for_repo(repo_name)
    finally:
        graph_db.close()

    console.print(
        Panel(
            f"Total nodes: [bold]{stats['nodes']}[/bold]  |  "
            f"Total edges: [bold]{stats['edges']}[/bold]  |  "
            f"Nodes for [cyan]{repo_name}[/cyan]: [bold]{len(nodes)}[/bold]",
            title=f"Graph: {repo_name}",
        )
    )

    type_counts = Counter(n.node_type for n in nodes)
    table = Table(title="Node types", show_header=True, header_style="bold magenta")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right", style="yellow")
    for node_type, count in type_counts.most_common():
        table.add_row(node_type, str(count))
    console.print(table)

    symbols = [n for n in nodes if n.node_type in ("function", "class")][:20]
    if symbols:
        sym_table = Table(
            title="Top symbols (first 20)", show_header=True, header_style="bold blue"
        )
        sym_table.add_column("Symbol", style="cyan")
        sym_table.add_column("Type", style="green")
        sym_table.add_column("File", style="dim")
        sym_table.add_column("Lines", justify="right")
        for n in symbols:
            lines = f"{n.start_line}-{n.end_line}" if n.start_line else ""
            sym_table.add_row(n.symbol_name or "", n.node_type, n.path, lines)
        console.print(sym_table)


@graph_group.command("impact")
@click.argument("file_path")
@click.option("--repo", "repo_name", default=None, help="Repo name.")
@click.option("--depth", default=5, show_default=True, help="Max traversal depth.")
def graph_impact(file_path: str, repo_name: str | None, depth: int) -> None:
    """Show what would be affected by changing a file.

    \b
        codecompass graph impact codecompass/retrieve/hybrid.py --repo code-compass
    """
    from codecompass.config.manager import load_config
    from codecompass.graph.model import GraphDB, make_node_id

    config = load_config()
    graph_db_path = config.store.resolved_data_dir() / "graph.db"

    if not graph_db_path.exists():
        console.print(f"[red]Graph database not found at {graph_db_path}.[/red]")
        sys.exit(1)

    norm_path = file_path.replace("\\", "/")
    repo = repo_name or norm_path.split("/")[0]

    graph_db = GraphDB(graph_db_path)
    try:
        node_id = make_node_id(repo, norm_path)
        impacted_ids = graph_db.impact_of_change(node_id, max_depth=depth)
        impacted_nodes = [graph_db.get_node(nid) for nid in impacted_ids]
        impacted_nodes = [n for n in impacted_nodes if n is not None]
    finally:
        graph_db.close()

    if not impacted_nodes:
        console.print(f"[dim]No dependents found for [cyan]{norm_path}[/cyan].[/dim]")
        return

    table = Table(
        title=f"Impact of changing {norm_path} ({len(impacted_nodes)} affected)",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("File", style="cyan")
    table.add_column("Symbol", style="yellow")
    table.add_column("Type", style="green")
    for n in impacted_nodes:
        table.add_row(n.path, n.symbol_name or "", n.node_type)
    console.print(table)


@graph_group.command("deps")
@click.argument("file_path")
@click.option("--repo", "repo_name", default=None, help="Repo name.")
@click.option("--depth", default=1, show_default=True, help="Dependency traversal depth.")
def graph_deps(file_path: str, repo_name: str | None, depth: int) -> None:
    """Show what a file depends on.

    \b
        codecompass graph deps codecompass/generate/answerer.py --repo code-compass
    """
    from codecompass.config.manager import load_config
    from codecompass.graph.model import GraphDB, make_node_id

    config = load_config()
    graph_db_path = config.store.resolved_data_dir() / "graph.db"

    if not graph_db_path.exists():
        console.print(f"[red]Graph database not found at {graph_db_path}.[/red]")
        sys.exit(1)

    norm_path = file_path.replace("\\", "/")
    repo = repo_name or norm_path.split("/")[0]

    graph_db = GraphDB(graph_db_path)
    try:
        node_id = make_node_id(repo, norm_path)
        dep_ids = graph_db.dependencies_of(node_id, depth=depth)
        dep_nodes = [graph_db.get_node(nid) for nid in dep_ids]
        dep_nodes = [n for n in dep_nodes if n is not None]
    finally:
        graph_db.close()

    if not dep_nodes:
        console.print(f"[dim]No dependencies found for [cyan]{norm_path}[/cyan].[/dim]")
        return

    table = Table(
        title=f"Dependencies of {norm_path} ({len(dep_nodes)} found)",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("File", style="cyan")
    table.add_column("Symbol", style="yellow")
    table.add_column("Type", style="green")
    for n in dep_nodes:
        table.add_row(n.path, n.symbol_name or "", n.node_type)
    console.print(table)


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------


@click.group("mcp")
def mcp_group() -> None:
    """MCP server commands."""


cli.add_command(mcp_group)


@mcp_group.command("serve")
def mcp_serve() -> None:
    """Start the code-compass MCP server (stdio transport).

    Add to your MCP client config:

    \b
    Claude Code (~/.claude/claude_desktop_config.json):
        {
          "mcpServers": {
            "code-compass": {
              "command": "codecompass",
              "args": ["mcp", "serve"]
            }
          }
        }

    \b
    Cursor (~/.cursor/mcp.json):
        {
          "mcpServers": {
            "code-compass": {
              "command": "codecompass",
              "args": ["mcp", "serve"]
            }
          }
        }
    """
    from mcp_server.server import main

    main()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
