from dataclasses import dataclass, field
import networkx as nx
import json
from pathlib import Path


@dataclass
class GraphNode:
    id: str          # "repo:path" or "repo:path:symbol"
    repo: str
    path: str
    symbol: str | None = None
    language: str | None = None
    node_type: str = "file"  # "file", "class", "function", "module"


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str  # "imports", "defines", "calls", "depends_on"


class DependencyGraph:
    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    def add_node(self, node: GraphNode) -> None:
        self._g.add_node(node.id, **{k: v for k, v in node.__dict__.items() if v is not None})

    def add_edge(self, edge: GraphEdge) -> None:
        self._g.add_edge(edge.source, edge.target, edge_type=edge.edge_type)

    def dependencies_of(self, node_id: str) -> list[str]:
        return list(self._g.successors(node_id))

    def dependents_of(self, node_id: str) -> list[str]:
        return list(self._g.predecessors(node_id))

    def impact_of_change(self, node_id: str, depth: int = 3) -> list[str]:
        """Return all nodes that transitively depend on node_id."""
        if node_id not in self._g:
            return []
        visited: set[str] = set()
        frontier = {node_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for n in frontier:
                for pred in self._g.predecessors(n):
                    if pred not in visited:
                        next_frontier.add(pred)
                        visited.add(pred)
            frontier = next_frontier
        return sorted(visited)

    def node_count(self) -> int:
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._g)
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self._g = nx.node_link_graph(data)
