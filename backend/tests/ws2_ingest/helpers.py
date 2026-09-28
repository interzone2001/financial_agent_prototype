"""Shared WS2 test data (importable, unlike conftest)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from chromadb import Documents, EmbeddingFunction, Embeddings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
UA = "WS2Tests test@example.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
ARCH = "https://www.sec.gov/Archives/edgar/data/320193"
ACC_10K, ACC_10Q, ACC_8K = "0000320193-25-000079", "0000320193-26-000020", "0000320193-26-000018"
URL_10K = f"{ARCH}/000032019325000079/aapl-20250927.htm"
URL_10Q = f"{ARCH}/000032019326000020/aapl-20260627.htm"
URL_8K = f"{ARCH}/000032019326000018/aapl-20260730.htm"
INDEX_8K = f"{ARCH}/000032019326000018/index.json"

# Synthetic 10-Q: TOC first; Part II Item 2 is LONGER than Part I Item 2 (MD&A), so a naive
# "longest span" pick would grab share repurchases instead of MD&A.
TEN_Q_HTML = (
    "<html><body>"
    "<p>PART I Item 1. Financial Statements 1 Item 2. Management's Discussion and Analysis 13 "
    "PART II Item 1A. Risk Factors 20 Item 2. Unregistered Sales of Equity Securities 21</p>"
    "<p>PART I Item 1. Financial Statements " + "Net sales table row. " * 40 + "</p>"
    "<p>Item 2. Management's Discussion and Analysis "
    + "Services net sales grew on strong demand. " * 60 + "</p>"
    "<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk None.</p>"
    "<p>PART II Item 1. Legal Proceedings None.</p>"
    "<p>Item 1A. Risk Factors " + "Tariffs and export controls may affect supply. " * 30 + "</p>"
    "<p>Item 2. Unregistered Sales of Equity Securities and Use of Proceeds "
    + "Share repurchases detail row. " * 120 + "</p>"
    "<p>Item 5. Other Information None.</p></body></html>"
)


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class HashEmbedding(EmbeddingFunction[Documents]):
    """Deterministic, offline 16-d embedding so tests never load the ONNX model."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in input]

    @staticmethod
    def name() -> str:
        return "ws2-test-hash"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> HashEmbedding:
        return HashEmbedding()
