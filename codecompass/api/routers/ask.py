from fastapi import APIRouter, Request
from codecompass.api.schemas import AskRequest, AskResponse, CitationSchema

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    answerer = request.app.state.answerer
    answer = answerer.answer(body.question, filters=body.filters)
    return AskResponse(
        answer=answer.text,
        citations=[
            CitationSchema(
                path=c.path,
                start_line=c.start_line,
                end_line=c.end_line,
                chunk_id=c.chunk_id,
            )
            for c in answer.citations
        ],
        question=answer.question,
        retrieved_chunk_count=len(answer.retrieved_chunk_ids),
    )
