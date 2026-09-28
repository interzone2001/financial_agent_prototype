import os

import pytest

# Tracing off in tests unless explicitly running live ones with it on.
os.environ.setdefault("LANGSMITH_TRACING", "false")
# create_app() with no deps must stay offline/keyless in tests.
os.environ.setdefault("APP_MODE", "stub")


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path_factory, monkeypatch):
    """Never write caches/checkpoints/Chroma into the repo's data/ during tests."""
    if not os.getenv("DATA_DIR_KEEP"):
        monkeypatch.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Isolated DATA_DIR for sqlite caches, checkpoints and Chroma."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_llm():
    from tests.fakes import FakeLLM

    return FakeLLM()
