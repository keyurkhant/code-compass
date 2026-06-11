from pathlib import Path
from typing import Iterator
import pathspec

SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", "dist", "build", ".venv", "venv",
    "env", ".env", "target", "vendor", ".tox", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "coverage", ".coverage", "htmlcov", "eggs", ".eggs",
    "buck-out", ".gradle", ".idea", ".vscode",
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".whl", ".egg", ".lock", ".sum", ".min.js", ".min.css",
    ".ttf", ".woff", ".woff2", ".eot", ".otf",
    ".mp4", ".mp3", ".wav", ".avi", ".mov",
    ".db", ".sqlite", ".sqlite3",
}

SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "Pipfile.lock", "poetry.lock",
    "Cargo.lock", "go.sum", "composer.lock", "Gemfile.lock",
}

MAX_FILE_SIZE_BYTES = 500_000  # 500KB


def _load_gitignore(repo_path: Path) -> pathspec.PathSpec | None:
    gitignore = repo_path / ".gitignore"
    if gitignore.exists():
        patterns = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    return None


def walk_repo(repo_path: Path) -> Iterator[Path]:
    """Yield text files in repo_path, skipping binaries, build dirs, and gitignored paths."""
    gitignore = _load_gitignore(repo_path)

    for path in sorted(repo_path.rglob("*")):
        if not path.is_file():
            continue

        # Check if any parent dir is in SKIP_DIRS
        rel = path.relative_to(repo_path)
        parts = rel.parts
        if any(part in SKIP_DIRS for part in parts[:-1]):
            continue

        # Skip by filename
        if path.name in SKIP_FILENAMES:
            continue

        # Skip by extension (handle compound extensions like .min.js)
        name_lower = path.name.lower()
        if any(name_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
            continue

        # Skip large files
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            continue

        # Respect .gitignore
        if gitignore and gitignore.match_file(str(rel)):
            continue

        # Quick binary check: read first 8KB and look for null bytes
        try:
            chunk = path.read_bytes()[:8192]
            if b"\x00" in chunk:
                continue
        except OSError:
            continue

        yield path
