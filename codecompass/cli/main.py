"""
code-compass CLI.

Commands:
  ingest  -- Walk a repo, chunk, embed, and index its source files.
  ask     -- Ask a question. Pass --repo to auto-ingest first.
  run     -- Ingest a repo then start the API server (one command, no fuss).
  eval    -- Run the retrieval evaluation harness against a golden set.
  serve   -- Start the FastAPI server.
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

console = Console()


def _build_providers(settings):
    """Instantiate the provider chain from settings — works with any LLM provider."""
    from codecompass.providers.embedding_fastembed import FastEmbedProvider
    from codecompass.providers.llm_litellm import LiteLLMProvider

    embedder = FastEmbedProvider(settings.embedding_model_name)
    llm = LiteLLMProvider(settings.llm_model)
    return embedder, llm


def _build_store(settings, collection_name: str):
    from codecompass.providers.vector_store_chroma import ChromaVectorStore

    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=collection_name,
    )


def _do_ingest(repo_path_or_url: str, repo_name: str | None, force: bool, settings) -> str:
    """Core ingest logic. Returns the collection name used."""
    from codecompass.index.bm25_indexer import BM25Index
    from codecompass.index.vector_indexer import index_chunks
    from codecompass.ingest.reader import ingest_repo

    embedder, _ = _build_providers(settings)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Walking repository and chunking files…", total=None)
        try:
            chunks, _ = ingest_repo(repo_path_or_url, repo_name=repo_name)
        except Exception as exc:
            console.print(f"[red]Ingest failed:[/red] {exc}")
            sys.exit(1)
        p.update(task, description=f"Chunked {len(chunks)} code chunks", completed=1, total=1)

    derived_name = repo_name or (chunks[0].repo if chunks else "default")
    collection_name = f"{settings.chroma_collection_prefix}_{derived_name}"
    console.print(f"Collection: [cyan]{collection_name}[/cyan]")

    store = _build_store(settings, collection_name)
    if force:
        console.print("[yellow]--force:[/yellow] rebuilding existing collection…")
        try:
            store.delete_collection(collection_name)
            store = _build_store(settings, collection_name)
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] {exc}")

    # Embed
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as p:
        task = p.add_task("Embedding chunks…", total=len(chunks))
        batch_size = settings.embedding_batch_size
        chunks_with_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.embed_text() for c in batch]
            embeddings = embedder.embed(texts)
            chunks_with_embeddings.extend(zip(batch, embeddings, strict=False))
            p.advance(task, len(batch))

    # Vector store
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Indexing into vector store…", total=None)
        count = index_chunks(chunks_with_embeddings, store)
        p.update(task, description=f"Indexed {count} chunks", completed=1, total=1)

    # BM25
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Building BM25 index…", total=None)
        bm25 = BM25Index()
        bm25.build(chunks)
        bm25_path = Path(settings.bm25_index_path)
        bm25.save(bm25_path)
        p.update(task, description=f"BM25 saved to {bm25_path}", completed=1, total=1)

    console.print(
        Panel(
            f"[green]Done![/green] {len(chunks)} chunks → collection [cyan]{collection_name}[/cyan]",
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
    """Ingest a local repo or remote git URL into the vector index."""
    from codecompass.config import get_settings

    _do_ingest(repo_path_or_url, repo_name, force, get_settings())


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


@cli.command("ask")
@click.argument("question")
@click.option(
    "--repo", "repo_path_or_url", default=None, help="Auto-ingest this repo first if not indexed."
)
@click.option("--repo-name", default=None, help="Repo name / collection prefix to query.")
@click.option("--top-k", default=10, show_default=True, help="Chunks to retrieve.")
@click.option("--filter-lang", default=None, help="Restrict to a language (python, go, …).")
@click.option("--filter-path", default=None, help="Restrict to paths containing this substring.")
def ask_cmd(
    question: str,
    repo_path_or_url: str | None,
    repo_name: str | None,
    top_k: int,
    filter_lang: str | None,
    filter_path: str | None,
) -> None:
    """Ask a question about the indexed codebase.

    Pass --repo to auto-ingest a repository before answering:

    \b
        codecompass ask "how does auth work?" --repo /path/to/repo
        codecompass ask "where is Session defined?" --repo https://github.com/psf/requests
    """
    from codecompass.config import get_settings
    from codecompass.generate.answerer import Answerer
    from codecompass.index.bm25_indexer import BM25Index

    settings = get_settings()

    # Auto-ingest if the caller provided a repo path
    if repo_path_or_url:
        collection_name = _do_ingest(repo_path_or_url, repo_name, force=False, settings=settings)
    else:
        derived = repo_name or "default"
        collection_name = f"{settings.chroma_collection_prefix}_{derived}"

    embedder, llm = _build_providers(settings)
    store = _build_store(settings, collection_name)

    bm25 = BM25Index()
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        bm25.load(bm25_path)
    else:
        console.print("[yellow]Warning:[/yellow] BM25 index not found; using dense-only retrieval.")

    # Build Chroma where-clause
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

    updated_settings = settings.model_copy(update={"top_k_retrieve": top_k})
    answerer = Answerer(
        llm=llm, embedder=embedder, store=store, bm25=bm25, settings=updated_settings
    )

    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")

    with console.status("[bold green]Thinking…[/bold green]", spinner="dots"):
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
    """Ingest a repo then start the API server — one command, no fuss.

    \b
        codecompass run /path/to/repo
        codecompass run https://github.com/psf/requests --port 8080
    """
    from codecompass.config import get_settings

    settings = get_settings()
    _do_ingest(repo_path_or_url, repo_name, force, settings)

    console.print(
        f"\n[bold green]Starting server[/bold green] on [cyan]http://{host}:{port}[/cyan]"
    )
    console.print("Open [cyan]streamlit run ui/app.py[/cyan] for the chat UI.\n")

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
    """Run the retrieval evaluation harness against a golden question set."""
    from codecompass.config import get_settings
    from codecompass.eval import metrics as _m
    from codecompass.eval.report import print_eval_table, save_eval_result
    from codecompass.eval.runner import EvalResult, PerQueryResult, load_golden
    from codecompass.generate.answerer import Answerer
    from codecompass.index.bm25_indexer import BM25Index
    from codecompass.retrieve.hybrid import hybrid_search

    settings = get_settings()
    embedder, llm = _build_providers(settings)
    derived = repo_name or "default"
    collection_name = f"{settings.chroma_collection_prefix}_{derived}"
    store = _build_store(settings, collection_name)

    bm25 = BM25Index()
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        bm25.load(bm25_path)

    answerer = Answerer(llm=llm, embedder=embedder, store=store, bm25=bm25, settings=settings)

    console.print(f"Loading golden set from [cyan]{golden_path}[/cyan]…")
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
        task = p.add_task("Running eval queries…", total=len(golden))
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
def serve_cmd(host: str, port: int) -> None:
    """Start the code-compass FastAPI server."""
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
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Server error:[/red] {exc}")
        sys.exit(exc.returncode)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
