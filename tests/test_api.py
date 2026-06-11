from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from codecompass.generate.models import Answer, Citation


@pytest.fixture
def mock_answerer():
    answerer = MagicMock()
    answerer.answer.return_value = Answer(
        text="The Session class is defined in `requests/sessions.py:100-200`.",
        citations=[
            Citation(path="requests/sessions.py", start_line=100, end_line=200, chunk_id="abc123")
        ],
        question="Where is Session defined?",
        retrieved_chunk_ids=["abc123"],
    )
    return answerer


@pytest.fixture
def client(mock_answerer):
    from codecompass.api.app import create_app

    @asynccontextmanager
    async def _stub_lifespan(app):
        app.state.answerer = mock_answerer
        yield

    with patch("codecompass.api.app.lifespan", _stub_lifespan):
        app = create_app()

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_valid(client, mock_answerer):
    response = client.post("/ask", json={"question": "Where is Session defined?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["citations"]) == 1


def test_ask_empty_question(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
