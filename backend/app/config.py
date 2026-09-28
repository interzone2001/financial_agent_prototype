"""Settings loaded from the repo-root `.env` (each worktree has a symlink to main's)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def data_dir() -> Path:
    p = Path(os.getenv("DATA_DIR", REPO_ROOT / "data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def alpha_vantage_key() -> str:
    return os.environ["ALPHA_VANTAGE_API_KEY"]


def sec_user_agent() -> str:
    return os.getenv("SEC_USER_AGENT", "FinancialAgentPrototype admin@example.com")
