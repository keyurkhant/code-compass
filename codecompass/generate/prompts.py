from codecompass.providers.base import SearchResult

SYSTEM_PROMPT = """You are a code understanding assistant. You answer questions about code repositories based ONLY on the retrieved code snippets provided to you.

Rules:
1. Only make claims that are directly supported by the provided code context.
2. Cite every claim using the format: `path/to/file.py:start_line-end_line`
3. If the context does not contain enough information to answer the question, say: "I don't have enough context to answer this question. The relevant code may not have been indexed or retrieved."
4. Never hallucinate function names, class names, or behavior not present in the context.
5. Be precise and technical. Your audience is software engineers."""


def build_context_block(results: list[SearchResult]) -> str:
    """Format retrieved chunks as a numbered context block."""
    parts = []
    for i, result in enumerate(results, 1):
        meta = result.metadata
        path = meta.get("path", "unknown")
        start = meta.get("start_line", "?")
        end = meta.get("end_line", "?")
        header = f"[{i}] {path}:{start}-{end}"
        parts.append(f"{header}\n```\n{result.document}\n```")
    return "\n\n".join(parts)


def build_user_message(context_block: str, question: str) -> str:
    return f"""Retrieved code context:

{context_block}

Question: {question}

Answer based only on the context above. Include citations in `path:start-end` format."""
