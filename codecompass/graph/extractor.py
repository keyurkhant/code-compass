import ast
import logging
from pathlib import Path
from codecompass.ingest.models import CodeChunk
from codecompass.graph.model import DependencyGraph, GraphEdge, GraphNode

logger = logging.getLogger(__name__)


def _extract_python_imports(source: str) -> list[str]:
    """Return list of module names imported in the source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return imports


def extract_dependencies(chunks: list[CodeChunk], repo_root: Path) -> DependencyGraph:
    """Build a dependency graph from code chunks."""
    graph = DependencyGraph()

    # Build a map of path -> node_id for quick lookup
    path_to_node: dict[str, str] = {}

    for chunk in chunks:
        node_id = f"{chunk.repo}:{chunk.path}"
        if node_id not in path_to_node:
            path_to_node[chunk.path] = node_id
            graph.add_node(GraphNode(
                id=node_id,
                repo=chunk.repo,
                path=chunk.path,
                language=chunk.language,
                node_type="file",
            ))

    # Extract imports for Python files
    seen_files: set[str] = set()
    for chunk in chunks:
        if chunk.language != "python" or chunk.path in seen_files:
            continue
        seen_files.add(chunk.path)

        file_path = repo_root / chunk.path
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        imports = _extract_python_imports(source)
        source_id = f"{chunk.repo}:{chunk.path}"

        for imported_module in imports:
            # Resolve relative to repo: check if any file matches
            candidates = [
                p for p in path_to_node
                if p.replace("/", ".").replace("\\", ".").endswith(imported_module)
                or p.endswith(f"{imported_module}.py")
                or p.endswith(f"{imported_module}/__init__.py")
            ]
            for candidate_path in candidates:
                target_id = path_to_node[candidate_path]
                if target_id != source_id:
                    graph.add_edge(GraphEdge(
                        source=source_id,
                        target=target_id,
                        edge_type="imports",
                    ))

    return graph
