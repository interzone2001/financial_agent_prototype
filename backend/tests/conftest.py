import os

import pytest

# Tracing off in tests unless explicitly running live ones with it on.
os.environ.setdefault("LANGSMITH_TRACING", "false")


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Isolated DATA_DIR for sqlite caches, checkpoints and Chroma."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_llm():
    from tests.fakes import FakeLLM

    return FakeLLM()
