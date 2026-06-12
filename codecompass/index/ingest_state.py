"""Tracks per-file SHA-256 hashes for incremental ingestion.

On re-ingest, only files whose hash changed are re-embedded.
DB lives at <data_dir>/ingest_hashes.db.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class IngestStateDB:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_hashes (
                repo      TEXT NOT NULL,
                path      TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                PRIMARY KEY (repo, path)
            )
            """
        )
        self._conn.commit()

    def get_hash(self, repo: str, path: str) -> str | None:
        row = self._conn.execute(
            "SELECT file_hash FROM file_hashes WHERE repo=? AND path=?", (repo, path)
        ).fetchone()
        return row[0] if row else None

    def set_hashes_bulk(self, repo: str, path_hashes: dict[str, str]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO file_hashes (repo, path, file_hash) VALUES (?,?,?)",
            [(repo, p, h) for p, h in path_hashes.items()],
        )
        self._conn.commit()

    def get_all_paths(self, repo: str) -> set[str]:
        rows = self._conn.execute("SELECT path FROM file_hashes WHERE repo=?", (repo,)).fetchall()
        return {r[0] for r in rows}

    def delete_paths(self, repo: str, paths: set[str]) -> None:
        self._conn.executemany(
            "DELETE FROM file_hashes WHERE repo=? AND path=?",
            [(repo, p) for p in paths],
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def compute_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
