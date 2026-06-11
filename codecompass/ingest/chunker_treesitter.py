"""AST-aware chunker using tree-sitter v0.23+ API.

Splits source files on function/class boundaries for Python, JavaScript, and Go.
Falls back gracefully when a grammar is unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from codecompass.ingest.models import CodeChunk
from codecompass.ingest.language import detect_language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node-type definitions per language
# ---------------------------------------------------------------------------

SYMBOL_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "method_definition",
        "class_declaration",
        "lexical_declaration",  # filtered further to arrow-function consts
    },
    "typescript": {
        "function_declaration",
        "method_definition",
        "class_declaration",
        "lexical_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
}

MAX_CHUNK_LINES = 150
MIN_CHUNK_LINES = 5


# ---------------------------------------------------------------------------
# Grammar loader
# ---------------------------------------------------------------------------

def _load_parser(language: str):
    """Return a (parser, Language) pair for the requested language.

    Uses the tree-sitter v0.23+ API:
        Language(grammar_module.language())
        Parser(language_obj)
    """
    from tree_sitter import Language, Parser  # type: ignore

    if language in ("python",):
        import tree_sitter_python as grammar  # type: ignore
        lang_obj = Language(grammar.language())

    elif language in ("javascript", "typescript"):
        # typescript grammar is separate but falls through to JS for chunking
        try:
            if language == "typescript":
                import tree_sitter_typescript as grammar  # type: ignore
                lang_obj = Language(grammar.language_typescript())
            else:
                import tree_sitter_javascript as grammar  # type: ignore
                lang_obj = Language(grammar.language())
        except (ImportError, AttributeError):
            import tree_sitter_javascript as grammar  # type: ignore
            lang_obj = Language(grammar.language())

    elif language == "go":
        import tree_sitter_go as grammar  # type: ignore
        lang_obj = Language(grammar.language())

    else:
        raise ValueError(f"Unsupported tree-sitter language: {language!r}")

    parser = Parser(lang_obj)
    return parser, lang_obj


# ---------------------------------------------------------------------------
# Symbol-name extraction helpers
# ---------------------------------------------------------------------------

def _get_child_by_field(node, field_name: str):
    """Return the first child node matching a field name."""
    for i in range(node.child_count):
        child = node.children[i]
        if node.field_name_for_child(i) == field_name:
            return child
    return None


def _extract_symbol_name(node, language: str) -> str | None:
    """Best-effort extraction of the declared symbol name from an AST node."""
    # Try field-based lookup first (most reliable)
    for field in ("name", "declarator"):
        child = _get_child_by_field(node, field)
        if child is not None:
            if child.type == "identifier":
                return child.text.decode("utf-8", errors="replace")
            # For JS lexical_declaration: var_decl -> identifier
            name_child = _get_child_by_field(child, "name")
            if name_child and name_child.type == "identifier":
                return name_child.text.decode("utf-8", errors="replace")

    # Fallback: first identifier child
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace")

    return None


def _is_arrow_function_const(node) -> bool:
    """Return True if a lexical_declaration node contains an arrow function."""
    for child in node.children:
        if child.type == "variable_declarator":
            for subchild in child.children:
                if subchild.type == "arrow_function":
                    return True
    return False


# ---------------------------------------------------------------------------
# Chunk splitting
# ---------------------------------------------------------------------------

def _split_large_chunk(
    lines: list[str],
    start_line: int,
    end_line: int,
    path: str,
    repo: str,
    language: str,
    symbol_name: str | None,
    context_prefix: str,
) -> list[CodeChunk]:
    """Split a node that exceeds MAX_CHUNK_LINES into two halves."""
    mid = (start_line + end_line) // 2
    chunks: list[CodeChunk] = []

    for part_start, part_end, suffix in [
        (start_line, mid, ""),
        (mid, end_line, " [continued]"),
    ]:
        content = "\n".join(lines[part_start - 1 : part_end])
        if not content.strip():
            continue
        chunk_id = CodeChunk.make_id(repo, path, part_start)
        chunks.append(
            CodeChunk(
                id=chunk_id,
                repo=repo,
                path=path,
                language=language,
                symbol_name=symbol_name,
                start_line=part_start,
                end_line=part_end,
                content=content,
                context_prefix=context_prefix + suffix,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Main chunker class
# ---------------------------------------------------------------------------

class TreeSitterChunker:
    """AST-aware chunker that splits on function/class boundaries."""

    def __init__(self, language: str) -> None:
        if language not in SYMBOL_NODE_TYPES:
            raise ValueError(
                f"Language {language!r} is not supported by TreeSitterChunker. "
                f"Supported: {sorted(SYMBOL_NODE_TYPES)}"
            )
        self.language = language
        self._parser, self._lang = _load_parser(language)
        self._node_types = SYMBOL_NODE_TYPES[language]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, source: str, path: str, repo: str) -> list[CodeChunk]:
        """Parse *source* and return a list of CodeChunk objects."""
        try:
            source_bytes = source.encode("utf-8")
        except UnicodeEncodeError:
            logger.warning("UnicodeEncodeError encoding source for %s; skipping", path)
            return []

        try:
            tree = self._parser.parse(source_bytes)
        except Exception as exc:
            logger.warning("tree-sitter parse error for %s: %s", path, exc)
            return []

        lines = source.splitlines()
        import_lines = self._extract_import_lines(lines)
        covered: list[tuple[int, int]] = []  # (start_line, end_line) 1-indexed

        chunks: list[CodeChunk] = []
        self._visit(
            tree.root_node,
            source_bytes,
            lines,
            path,
            repo,
            import_lines,
            enclosing_class=None,
            chunks=chunks,
            covered=covered,
        )

        # Module-level preamble (before first symbol)
        chunks.extend(
            self._preamble_chunk(lines, covered, path, repo, import_lines)
        )

        return chunks

    # ------------------------------------------------------------------
    # AST traversal
    # ------------------------------------------------------------------

    def _visit(
        self,
        node,
        source_bytes: bytes,
        lines: list[str],
        path: str,
        repo: str,
        import_lines: str,
        enclosing_class: str | None,
        chunks: list[CodeChunk],
        covered: list[tuple[int, int]],
    ) -> None:
        node_type = node.type

        if node_type in self._node_types:
            # Filter JS lexical_declaration to arrow-function consts only
            if node_type == "lexical_declaration" and not _is_arrow_function_const(node):
                for child in node.children:
                    self._visit(
                        child, source_bytes, lines, path, repo,
                        import_lines, enclosing_class, chunks, covered,
                    )
                return

            start_line = node.start_point[0] + 1  # 1-indexed
            end_line = node.end_point[0] + 1

            node_lines = end_line - start_line + 1
            if node_lines < MIN_CHUNK_LINES:
                # Still recurse for nested definitions
                for child in node.children:
                    self._visit(
                        child, source_bytes, lines, path, repo,
                        import_lines, enclosing_class, chunks, covered,
                    )
                return

            symbol_name = _extract_symbol_name(node, self.language)
            context_prefix = self._build_context_prefix(
                path, symbol_name, enclosing_class, import_lines
            )

            if node_lines > MAX_CHUNK_LINES:
                new_chunks = _split_large_chunk(
                    lines, start_line, end_line, path, repo,
                    self.language, symbol_name, context_prefix,
                )
            else:
                content = "\n".join(lines[start_line - 1 : end_line])
                chunk_id = CodeChunk.make_id(repo, path, start_line)
                new_chunks = [
                    CodeChunk(
                        id=chunk_id,
                        repo=repo,
                        path=path,
                        language=self.language,
                        symbol_name=symbol_name,
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        context_prefix=context_prefix,
                    )
                ]

            chunks.extend(new_chunks)
            covered.append((start_line, end_line))

            # Recurse into class bodies to also extract methods
            new_enclosing = symbol_name if node_type in (
                "class_definition", "class_declaration"
            ) else enclosing_class

            for child in node.children:
                self._visit(
                    child, source_bytes, lines, path, repo,
                    import_lines, new_enclosing, chunks, covered,
                )

        else:
            for child in node.children:
                self._visit(
                    child, source_bytes, lines, path, repo,
                    import_lines, enclosing_class, chunks, covered,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_import_lines(self, lines: list[str]) -> str:
        """Return the first 10 lines of the file as a string (captures imports)."""
        return "\n".join(lines[:10])

    def _build_context_prefix(
        self,
        path: str,
        symbol_name: str | None,
        enclosing_class: str | None,
        import_lines: str,
    ) -> str:
        parts = [f"# File: {path}"]
        if import_lines.strip():
            parts.append(import_lines)
        if enclosing_class:
            parts.append(f"# In class: {enclosing_class}")
        if symbol_name:
            parts.append(f"# Symbol: {symbol_name}")
        return "\n".join(parts)

    def _preamble_chunk(
        self,
        lines: list[str],
        covered: list[tuple[int, int]],
        path: str,
        repo: str,
        import_lines: str,
    ) -> list[CodeChunk]:
        """Return a chunk for module-level code before the first symbol, if significant."""
        if not lines:
            return []

        # Find the first covered line
        first_covered = min((s for s, _ in covered), default=len(lines) + 1)
        preamble_end = first_covered - 1  # last line before first symbol

        if preamble_end < MIN_CHUNK_LINES:
            return []

        content = "\n".join(lines[:preamble_end])
        if not content.strip():
            return []

        context_prefix = f"# File: {path}\n# Module-level preamble"
        chunk_id = CodeChunk.make_id(repo, path, 1)
        return [
            CodeChunk(
                id=chunk_id,
                repo=repo,
                path=path,
                language=self.language,
                symbol_name=None,
                start_line=1,
                end_line=preamble_end,
                content=content,
                context_prefix=context_prefix,
            )
        ]
