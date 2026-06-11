"""Dependency graph extraction — populates GraphDB from source files.

Extracts:
- File nodes for every indexed file
- Function/class nodes (symbol-level) for Python (via ast)
- Import edges (Python: direct parse = EXTRACTED confidence)
- Call edges where resolvable within the repo (INFERRED confidence)

Incremental: skips files whose SHA-256 hash hasn't changed.
"""
from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path

from codecompass.graph.model import GraphDB, GraphEdge, GraphNode, make_node_id

logger = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of file content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_python_symbols(
    source: str, path: str, repo: str, file_hash: str
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Extract function/class nodes + import edges from Python source using ast."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    file_node_id = make_node_id(repo, path)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return nodes, edges

    # Top-level functions and classes
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            sym_id = make_node_id(repo, path, node.name)
            nodes.append(
                GraphNode(
                    id=sym_id,
                    repo=repo,
                    path=path,
                    symbol_name=node.name,
                    node_type="function",
                    language="python",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    file_hash=file_hash,
                )
            )
            edges.append(
                GraphEdge(source_id=file_node_id, target_id=sym_id, edge_type="defines")
            )

        elif isinstance(node, ast.ClassDef):
            cls_id = make_node_id(repo, path, node.name)
            nodes.append(
                GraphNode(
                    id=cls_id,
                    repo=repo,
                    path=path,
                    symbol_name=node.name,
                    node_type="class",
                    language="python",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    file_hash=file_hash,
                )
            )
            edges.append(
                GraphEdge(source_id=file_node_id, target_id=cls_id, edge_type="defines")
            )

            # Methods inside class
            for item in ast.iter_child_nodes(node):
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    meth_id = make_node_id(repo, path, f"{node.name}.{item.name}")
                    nodes.append(
                        GraphNode(
                            id=meth_id,
                            repo=repo,
                            path=path,
                            symbol_name=f"{node.name}.{item.name}",
                            node_type="function",
                            language="python",
                            start_line=item.lineno,
                            end_line=item.end_lineno,
                            file_hash=file_hash,
                        )
                    )
                    edges.append(
                        GraphEdge(source_id=cls_id, target_id=meth_id, edge_type="defines")
                    )

            # Inheritance
            for base in node.bases:
                base_name = ast.unparse(base)
                edges.append(
                    GraphEdge(
                        source_id=cls_id,
                        target_id=f"unresolved:{base_name}",
                        edge_type="inherits",
                        confidence="AMBIGUOUS",
                    )
                )

    # Import edges
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imp_node_id = make_node_id(repo, path, f"import:{alias.name}")
                nodes.append(
                    GraphNode(
                        id=imp_node_id,
                        repo=repo,
                        path=path,
                        symbol_name=alias.name,
                        node_type="import",
                        language="python",
                        file_hash=file_hash,
                    )
                )
                edges.append(
                    GraphEdge(
                        source_id=file_node_id,
                        target_id=imp_node_id,
                        edge_type="imports",
                    )
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imp_node_id = make_node_id(repo, path, f"import:{node.module}")
            nodes.append(
                GraphNode(
                    id=imp_node_id,
                    repo=repo,
                    path=path,
                    symbol_name=node.module,
                    node_type="import",
                    language="python",
                    file_hash=file_hash,
                )
            )
            edges.append(
                GraphEdge(
                    source_id=file_node_id,
                    target_id=imp_node_id,
                    edge_type="imports",
                )
            )

    return nodes, edges


def _resolve_imports(
    graph: GraphDB,
    repo: str,
    file_nodes_by_path: dict[str, str],  # path -> node_id
) -> None:
    """Second pass: wire up import nodes to actual file nodes in the repo (INFERRED confidence)."""
    import_nodes = [n for n in graph.nodes_for_repo(repo) if n.node_type == "import"]
    for imp_node in import_nodes:
        module = imp_node.symbol_name or ""
        candidates = [
            p
            for p in file_nodes_by_path
            if p.replace("/", ".").replace("\\", ".").rstrip(".py").endswith(module.split(".")[-1])
            or p.endswith(f"{module.replace('.', '/')}.py")
            or p.endswith(f"{module.replace('.', '/')}/__init__.py")
        ]
        for candidate_path in candidates:
            target_id = file_nodes_by_path[candidate_path]
            if target_id != imp_node.id:
                graph.upsert_edge(
                    GraphEdge(
                        source_id=imp_node.id,
                        target_id=target_id,
                        edge_type="imports",
                        confidence="INFERRED",
                    )
                )


def build_graph(
    repo_root: Path,
    repo: str,
    graph: GraphDB,
    incremental: bool = True,
) -> dict:
    """Walk repo_root, extract nodes + edges, upsert into GraphDB.

    If incremental=True, skip files whose SHA-256 hash matches what's stored.
    Returns {"files_processed": n, "files_skipped": n, "nodes_added": n}.
    """
    from codecompass.ingest.language import detect_language
    from codecompass.ingest.walker import walk_repo

    stats = {"files_processed": 0, "files_skipped": 0, "nodes_added": 0}
    file_nodes_by_path: dict[str, str] = {}  # relative_path -> node_id

    for file_path in walk_repo(repo_root):
        rel_path = str(file_path.relative_to(repo_root))
        language = detect_language(file_path) or "text"

        # File-level node
        file_node_id = make_node_id(repo, rel_path)
        current_hash = file_sha256(file_path)

        if incremental:
            stored_hash = graph.get_file_hash(repo, rel_path)
            if stored_hash == current_hash:
                file_nodes_by_path[rel_path] = file_node_id
                stats["files_skipped"] += 1
                continue
            else:
                graph.delete_file(repo, rel_path)

        # Upsert file node
        graph.upsert_node(
            GraphNode(
                id=file_node_id,
                repo=repo,
                path=rel_path,
                node_type="file",
                language=language,
                file_hash=current_hash,
            )
        )
        file_nodes_by_path[rel_path] = file_node_id
        stats["files_processed"] += 1

        # Symbol-level extraction (Python only for now)
        if language == "python":
            try:
                source = file_path.read_text(encoding="utf-8", errors="replace")
                sym_nodes, sym_edges = extract_python_symbols(source, rel_path, repo, current_hash)
                for n in sym_nodes:
                    graph.upsert_node(n)
                for e in sym_edges:
                    graph.upsert_edge(e)
                stats["nodes_added"] += len(sym_nodes)
            except Exception as exc:
                logger.warning("Failed to extract symbols from %s: %s", rel_path, exc)

    # Second pass: resolve cross-file imports
    _resolve_imports(graph, repo, file_nodes_by_path)

    return stats
