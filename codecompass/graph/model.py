"""SQLite-backed dependency graph for code-compass.

Replaces the previous NetworkX in-memory graph with a persistent SQLite store.
Supports: typed edges with confidence, BFS/DFS via recursive CTEs, FTS5 keyword search,
incremental re-indexing by file SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GraphNode:
    id: str
    repo: str
    path: str
    symbol_name: str | None = None
    node_type: str = "file"  # file | function | class | import
    language: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    file_hash: str | None = None
    summary: str | None = None


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: str  # imports | calls | inherits | defines | depends_on
    confidence: str = "EXTRACTED"  # EXTRACTED | INFERRED | AMBIGUOUS


def make_node_id(repo: str, path: str, symbol: str | None = None) -> str:
    key = f"{repo}:{path}:{symbol or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class GraphDB:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

    def _apply_schema(self) -> None:
        # Read schema.sql from the same package directory
        schema_path = Path(__file__).parent / "schema.sql"
        self._conn.executescript(schema_path.read_text())
        self._conn.commit()

    def upsert_node(self, node: GraphNode) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO nodes
               (id, repo, path, symbol_name, node_type, language, start_line, end_line, file_hash, summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.id,
                node.repo,
                node.path,
                node.symbol_name,
                node.node_type,
                node.language,
                node.start_line,
                node.end_line,
                node.file_hash,
                node.summary,
            ),
        )
        # Keep FTS5 in sync
        self._conn.execute("DELETE FROM nodes_fts WHERE id = ?", (node.id,))
        self._conn.execute(
            "INSERT INTO nodes_fts (id, symbol_name, path, node_type, content) VALUES (?, ?, ?, ?, ?)",
            (node.id, node.symbol_name or "", node.path, node.node_type, node.summary or ""),
        )
        self._conn.commit()

    def upsert_edge(self, edge: GraphEdge) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO edges (source_id, target_id, edge_type, confidence)
               VALUES (?, ?, ?, ?)""",
            (edge.source_id, edge.target_id, edge.edge_type, edge.confidence),
        )
        self._conn.commit()

    def get_node(self, node_id: str) -> GraphNode | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return _row_to_node(row) if row else None

    def get_file_hash(self, repo: str, path: str) -> str | None:
        """Return the stored file_hash for (repo, path), or None if not indexed."""
        row = self._conn.execute(
            "SELECT file_hash FROM nodes WHERE repo = ? AND path = ? AND node_type = 'file' LIMIT 1",
            (repo, path),
        ).fetchone()
        return row["file_hash"] if row else None

    def delete_file(self, repo: str, path: str) -> None:
        """Remove all nodes and edges for a given file (for incremental re-index)."""
        node_ids = [
            r["id"]
            for r in self._conn.execute(
                "SELECT id FROM nodes WHERE repo = ? AND path = ?", (repo, path)
            )
        ]
        for nid in node_ids:
            self._conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (nid, nid))
            self._conn.execute("DELETE FROM nodes_fts WHERE id = ?", (nid,))
        self._conn.execute("DELETE FROM nodes WHERE repo = ? AND path = ?", (repo, path))
        self._conn.commit()

    def impact_of_change(self, node_id: str, max_depth: int = 5) -> list[str]:
        """Return node IDs that transitively depend on (import/call) node_id using BFS via recursive CTE."""
        rows = self._conn.execute(
            """
            WITH RECURSIVE impacted(id, depth) AS (
                SELECT ?, 0
                UNION
                SELECT e.source_id, i.depth + 1
                FROM edges e
                JOIN impacted i ON e.target_id = i.id
                WHERE i.depth < ?
                  AND e.edge_type IN ('imports', 'calls', 'inherits', 'depends_on')
            )
            SELECT DISTINCT id FROM impacted WHERE id != ?
            """,
            (node_id, max_depth, node_id),
        ).fetchall()
        return [r["id"] for r in rows]

    def dependencies_of(self, node_id: str, depth: int = 1) -> list[str]:
        """What does node_id depend on (outgoing edges)?"""
        rows = self._conn.execute(
            """
            WITH RECURSIVE deps(id, depth) AS (
                SELECT ?, 0
                UNION
                SELECT e.target_id, d.depth + 1
                FROM edges e
                JOIN deps d ON e.source_id = d.id
                WHERE d.depth < ?
            )
            SELECT DISTINCT id FROM deps WHERE id != ?
            """,
            (node_id, depth, node_id),
        ).fetchall()
        return [r["id"] for r in rows]

    def dependents_of(self, node_id: str) -> list[str]:
        """What directly imports/calls node_id (one hop)?"""
        rows = self._conn.execute(
            "SELECT DISTINCT source_id FROM edges WHERE target_id = ?", (node_id,)
        ).fetchall()
        return [r["source_id"] for r in rows]

    def fts_search(self, query: str, limit: int = 20) -> list[GraphNode]:
        """FTS5 keyword search over node symbol names, paths, content."""
        rows = self._conn.execute(
            """
            SELECT n.* FROM nodes n
            JOIN nodes_fts f ON n.id = f.id
            WHERE nodes_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def nodes_for_repo(self, repo: str) -> list[GraphNode]:
        rows = self._conn.execute("SELECT * FROM nodes WHERE repo = ?", (repo,)).fetchall()
        return [_row_to_node(r) for r in rows]

    def stats(self) -> dict:
        n = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        e = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return {"nodes": n, "edges": e}

    def close(self) -> None:
        self._conn.close()


def _row_to_node(row: sqlite3.Row) -> GraphNode:
    return GraphNode(
        id=row["id"],
        repo=row["repo"],
        path=row["path"],
        symbol_name=row["symbol_name"],
        node_type=row["node_type"],
        language=row["language"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        file_hash=row["file_hash"],
        summary=row["summary"],
    )
