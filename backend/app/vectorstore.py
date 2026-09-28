"""Chroma persistence for filing chunks (spec §3.5). Embedding fn is injectable for tests."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

import chromadb
from chromadb.api.models.Collection import Collection

from app import config
from app.models import Chunk, SourceRef

COLLECTION_NAME = "filings"


class ChunkRecord(NamedTuple):
    id: str  # chunk_id "{accession_no}:{section_slug}:{idx}"
    text: str
    metadata: dict[str, str | int]  # §3.5 keys (+ retrieved_at ISO), all scalar


def get_collection(data_dir: Path | None = None, embedding_function=None) -> Collection:
    """`filings` collection under DATA_DIR/chroma. None => Chroma's default ONNX MiniLM."""
    client = chromadb.PersistentClient(path=str((data_dir or config.data_dir()) / "chroma"))
    if embedding_function is None:
        return client.get_or_create_collection(COLLECTION_NAME)
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=embedding_function)


def warm_embedding_model() -> None:
    """Downloads (~80MB, once) and loads the default embedding model."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    DefaultEmbeddingFunction()(["warm up"])


def has_accession(collection: Collection, accession_no: str) -> bool:
    return bool(collection.get(where={"accession_no": accession_no}, limit=1)["ids"])


def add_records(collection: Collection, records: list[ChunkRecord]) -> int:
    if not records:  # Chroma rejects empty upserts
        return 0
    collection.upsert(ids=[r.id for r in records], documents=[r.text for r in records],
                      metadatas=[r.metadata for r in records])
    return len(records)


def _where(cik: str, form_type: str | None) -> dict:
    if form_type is None:
        return {"cik": cik}
    return {"$and": [{"cik": cik}, {"form_type": form_type}]}


def query_chunks(collection: Collection, cik: str, ticker: str, query: str, k: int = 6,
                 form_type: str | None = None) -> list[Chunk]:
    """Filters by CIK (shared across a company's ticker/share classes); `ticker` is stamped
    onto the returned Chunks as the requested ticker, not the one stored at ingest time."""
    if k < 1:
        return []
    res = collection.query(query_texts=[query], n_results=k, where=_where(cik, form_type))
    return [_to_chunk(cid, doc, meta, ticker)
            for cid, doc, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])]


def _to_chunk(chunk_id: str, text: str, m: dict, ticker: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, ticker=ticker, text=text, source=SourceRef(
        provider="sec_edgar", url=m["url"], retrieved_at=datetime.fromisoformat(m["retrieved_at"]),
        accession_no=m["accession_no"], form_type=m["form_type"],
        filed_date=date.fromisoformat(m["filed_date"]), section=m["section"], chunk_id=chunk_id))
