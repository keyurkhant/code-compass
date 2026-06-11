"""
code-compass CLI.

Commands:
  ingest  -- Walk a repo, chunk, embed, and index its source files.
  ask     -- Ask a question about an indexed codebase.
  eval    -- Run the retrieval evaluation harness against a golden set.
  serve   -- Start the FastAPI server.
"""

import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()


def _build_providers(settings):
    """Instantiate the standard provider chain from settings."""
    from codecompass.providers.embedding_sentence_transformer import SentenceTransformerEmbeddingProvider
    from codecompass.providers.llm_anthropic import AnthropicLLMProvider
    from codecompass.providers.vector_store_chroma import ChromaVectorStore

    embedder = SentenceTransformerEmbeddingProvider(settings.embedding_model_name)
    llm = AnthropicLLMProvider(
        model_name=settings.llm_model_name,
        api_key=settings.anthropic_api_key,
    )
    return embedder, llm


def _build_store(settings, collection_name: str):
    from codecompass.providers.vector_store_chroma import ChromaVectorStore

    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=collection_name,
    )


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """code-compass: RAG-powered code search and Q&A."""


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command("ingest")
@click.argument("repo_path_or_url")
@click.option("--name", "repo_name", default=None, help="Override the repo name used as collection prefix.")
@click.option("--force/--no-force", default=False, help="Delete and rebuild the existing collection.")
def ingest_cmd(repo_path_or_url: str, repo_name: str | None, force: bool) -> None:
    """Ingest a local repo or remote git URL into the vector index."""
    from codecompass.config import get_settings
    from codecompass.index.bm25_indexer import BM25Index
    from codecompass.index.embedder import embed_chunks
    from codecompass.index.vector_indexer import index_chunks
    from codecompass.ingest.reader import ingest_repo

    settings = get_settings()
    embedder, _ = _build_providers(settings)

    console.print(f"[bold cyan]Ingesting[/bold cyan] [green]{repo_path_or_url}[/green]")

    # --- step 1: walk + chunk ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Walking repository and chunking files...", total=None)
        try:
            chunks, repo_root = ingest_repo(repo_path_or_url, repo_name=repo_name)
        except Exception as exc:
            console.print(f"[red]Ingest failed:[/red] {exc}")
            sys.exit(1)
        progress.update(task, description=f"Chunked {len(chunks)} code chunks", completed=1, total=1)

    derived_name = repo_name or chunks[0].repo if chunks else "unknown"
    collection_name = f"{settings.chroma_collection_prefix}_{derived_name}"
    console.print(f"Collection: [cyan]{collection_name}[/cyan]")

    store = _build_store(settings, collection_name)

    # --- step 2: optionally delete existing collection ---
    if force:
        console.print("[yellow]--force:[/yellow] deleting existing collection...")
        try:
            store.delete_collection(collection_name)
            # Re-create after deletion
            store = _build_store(settings, collection_name)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] could not delete collection: {exc}")

    # --- step 3: embed ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(chunks))
        batch_size = settings.embedding_batch_size
        chunks_with_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            from codecompass.providers.base import EmbeddingProvider
            texts = [c.embed_text() for c in batch]
            embeddings = embedder.embed(texts)
            chunks_with_embeddings.extend(zip(batch, embeddings))
            progress.advance(task, len(batch))

    # --- step 4: index into vector store ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing into vector store...", total=None)
        count = index_chunks(chunks_with_embeddings, store)
        progress.update(task, description=f"Indexed {count} chunks", completed=1, total=1)

    # --- step 5: build and save BM25 index ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building BM25 index...", total=None)
        bm25 = BM25Index()
        bm25.build(chunks)
        bm25_path = Path(settings.bm25_index_path)
        bm25.save(bm25_path)
        progress.update(
            task,
            description=f"BM25 index saved to {bm25_path}",
            completed=1,
            total=1,
        )

    console.print(
        Panel(
            f"[green]Done![/green] Ingested [bold]{len(chunks)}[/bold] chunks from "
            f"[bold]{repo_path_or_url}[/bold] into collection [cyan]{collection_name}[/cyan].",
            title="Ingest complete",
        )
    )


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

@cli.command("ask")
@click.argument("question")
@click.option("--top-k", default=10, show_default=True, help="Number of chunks to retrieve.")
@click.option("--filter-lang", default=None, help="Restrict retrieval to a specific language.")
@click.option("--filter-path", default=None, help="Restrict retrieval to paths containing this substring.")
@click.option("--repo", "repo_name", default=None, help="Target a specific ingested repo collection.")
def ask_cmd(
    question: str,
    top_k: int,
    filter_lang: str | None,
    filter_path: str | None,
    repo_name: str | None,
) -> None:
    """Ask a question about the indexed codebase."""
    from codecompass.config import get_settings
    from codecompass.generate.answerer import Answerer
    from codecompass.index.bm25_indexer import BM25Index

    settings = get_settings()
    embedder, llm = _build_providers(settings)

    collection_name = (
        f"{settings.chroma_collection_prefix}_{repo_name}"
        if repo_name
        else f"{settings.chroma_collection_prefix}_default"
    )
    store = _build_store(settings, collection_name)

    bm25 = BM25Index()
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        bm25.load(bm25_path)
    else:
        console.print("[yellow]Warning:[/yellow] BM25 index not found; falling back to dense-only retrieval.")

    # Build where-clause filters
    filters: dict | None = None
    where_clauses = []
    if filter_lang:
        where_clauses.append({"language": {"$eq": filter_lang}})
    if filter_path:
        where_clauses.append({"path": {"$contains": filter_path}})
    if len(where_clauses) == 1:
        filters = where_clauses[0]
    elif len(where_clauses) > 1:
        filters = {"$and": where_clauses}

    # Override top_k from CLI
    settings_copy = settings.model_copy(update={"top_k_retrieve": top_k})
    answerer = Answerer(llm=llm, embedder=embedder, store=store, bm25=bm25, settings=settings_copy)

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
        table.add_column("Chunk ID", style="dim")
        for c in answer.citations:
            table.add_row(c.path, f"{c.start_line}-{c.end_line}", c.chunk_id)
        console.print(table)
    else:
        console.print("[dim]No citations extracted.[/dim]")


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
@click.option(
    "--k",
    "k_values",
    multiple=True,
    type=int,
    default=[1, 5, 10],
    show_default=True,
    help="k values for Recall@k (can be specified multiple times).",
)
@click.option(
    "--output-dir",
    "output_dir",
    default="./data/eval_results",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory where the JSON result file is saved.",
)
@click.option("--repo", "repo_name", default=None, help="Target a specific ingested repo collection.")
def eval_cmd(
    golden_path: Path,
    k_values: tuple[int, ...],
    output_dir: Path,
    repo_name: str | None,
) -> None:
    """Run the retrieval evaluation harness against a golden question set."""
    from codecompass.config import get_settings
    from codecompass.eval.report import print_eval_table, save_eval_result
    from codecompass.eval.runner import load_golden, run_eval
    from codecompass.generate.answerer import Answerer
    from codecompass.index.bm25_indexer import BM25Index

    settings = get_settings()
    embedder, llm = _build_providers(settings)

    collection_name = (
        f"{settings.chroma_collection_prefix}_{repo_name}"
        if repo_name
        else f"{settings.chroma_collection_prefix}_default"
    )
    store = _build_store(settings, collection_name)

    bm25 = BM25Index()
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        bm25.load(bm25_path)
    else:
        console.print("[yellow]Warning:[/yellow] BM25 index not found; falling back to dense-only retrieval.")

    answerer = Answerer(llm=llm, embedder=embedder, store=store, bm25=bm25, settings=settings)

    console.print(f"Loading golden set from [cyan]{golden_path}[/cyan]...")
    golden = load_golden(golden_path)
    console.print(f"Loaded [bold]{len(golden)}[/bold] questions.")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running eval queries...", total=len(golden))

        # Monkey-patch run_eval to show progress
        from codecompass.eval import metrics as _m
        from codecompass.eval.runner import PerQueryResult
        from codecompass.retrieve.hybrid import hybrid_search

        per_query = []
        _k_list = list(k_values)
        for entry in golden:
            results = hybrid_search(
                query=entry.question,
                provider=answerer._embedder,
                store=answerer._store,
                bm25_index=answerer._bm25,
                top_k=max(_k_list),
            )
            retrieved_paths = [r.metadata.get("path", "") for r in results]
            pq = PerQueryResult(
                question=entry.question,
                relevant_paths=entry.relevant_paths,
                retrieved_paths=retrieved_paths,
                recall_at_1=_m.recall_at_k(retrieved_paths, entry.relevant_paths, 1),
                recall_at_5=_m.recall_at_k(retrieved_paths, entry.relevant_paths, 5),
                recall_at_10=_m.recall_at_k(retrieved_paths, entry.relevant_paths, 10),
                rr=_m.reciprocal_rank(retrieved_paths, entry.relevant_paths),
            )
            per_query.append(pq)
            progress.advance(task)

    from codecompass.eval.runner import EvalResult
    result = EvalResult(
        per_query=per_query,
        recall_at_1=sum(pq.recall_at_1 for pq in per_query) / len(per_query) if per_query else 0.0,
        recall_at_5=sum(pq.recall_at_5 for pq in per_query) / len(per_query) if per_query else 0.0,
        recall_at_10=sum(pq.recall_at_10 for pq in per_query) / len(per_query) if per_query else 0.0,
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
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind the server to.")
@click.option("--port", default=8000, show_default=True, help="Port to listen on.")
def serve_cmd(host: str, port: int) -> None:
    """Start the code-compass FastAPI server with uvicorn."""
    console.print(
        f"[bold green]Starting server[/bold green] on [cyan]http://{host}:{port}[/cyan]"
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
        console.print(f"[red]Server exited with error:[/red] {exc}")
        sys.exit(exc.returncode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli()


if __name__ == "__main__":
    main()
