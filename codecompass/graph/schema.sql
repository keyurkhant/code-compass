-- Nodes: any symbol or file in the codebase
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,              -- sha256(repo:path:symbol)[:16]
    repo TEXT NOT NULL,
    path TEXT NOT NULL,
    symbol_name TEXT,                 -- NULL for file-level nodes
    node_type TEXT NOT NULL,          -- file | function | class | import
    language TEXT,
    start_line INTEGER,
    end_line INTEGER,
    file_hash TEXT,                   -- SHA-256 of file content (for incremental re-index)
    summary TEXT,                     -- optional LLM-generated summary
    indexed_at TEXT DEFAULT (datetime('now'))
);

-- Edges: relationships between nodes
CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,          -- imports | calls | inherits | defines | depends_on
    confidence TEXT NOT NULL DEFAULT 'EXTRACTED',  -- EXTRACTED | INFERRED | AMBIGUOUS
    PRIMARY KEY (source_id, target_id, edge_type),
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);

-- FTS5 for full-text keyword search over node metadata + content
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    id UNINDEXED,
    symbol_name,
    path,
    node_type UNINDEXED,
    content
);

-- Indexes for fast traversal
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_nodes_path ON nodes(repo, path);
CREATE INDEX IF NOT EXISTS idx_nodes_symbol ON nodes(repo, symbol_name);
