from dataclasses import dataclass, field


@dataclass
class Citation:
    path: str
    start_line: int
    end_line: int
    chunk_id: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    question: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
