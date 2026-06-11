import logging
import tempfile
from pathlib import Path

import git

from codecompass.ingest.chunker import chunk_file
from codecompass.ingest.models import CodeChunk
from codecompass.ingest.walker import walk_repo

logger = logging.getLogger(__name__)


def ingest_repo(
    repo_path_or_url: str, repo_name: str | None = None
) -> tuple[list[CodeChunk], Path]:
    """
    Ingest a local path or remote git URL.
    Returns (chunks, repo_root_path).
    If URL, clones to a temp dir (caller is responsible for cleanup).
    """
    if repo_path_or_url.startswith(("http://", "https://", "git@")):
        tmp = Path(tempfile.mkdtemp(prefix="codecompass_"))
        logger.info(f"Cloning {repo_path_or_url} to {tmp}")
        git.Repo.clone_from(repo_path_or_url, tmp, depth=1)
        repo_root = tmp
        derived_name = repo_path_or_url.rstrip("/").split("/")[-1].removesuffix(".git")
    else:
        repo_root = Path(repo_path_or_url).resolve()
        derived_name = repo_root.name

    name = repo_name or derived_name
    all_chunks: list[CodeChunk] = []

    files = list(walk_repo(repo_root))
    logger.info(f"Walking {len(files)} files in {repo_root}")

    for file_path in files:
        chunks = chunk_file(file_path, repo_root, name)
        all_chunks.extend(chunks)

    logger.info(f"Produced {len(all_chunks)} chunks from {len(files)} files")
    return all_chunks, repo_root
