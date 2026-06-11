from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TINY_REPO = FIXTURES_DIR / "tiny_repo"


@pytest.fixture
def tiny_repo_path() -> Path:
    return TINY_REPO
