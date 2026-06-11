from pathlib import Path

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".sh": "shell",
    ".bash": "shell",
}

TREESITTER_SUPPORTED = {"python", "javascript", "typescript", "go"}


def detect_language(path: Path) -> str | None:
    return EXTENSION_MAP.get(path.suffix.lower())
