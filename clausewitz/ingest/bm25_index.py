"""Ingestion step 5: build and load the BM25 keyword index over all chunks.

The index lives on disk (`BM25_INDEX_PATH`; Cloud Storage in prod) because Postgres has
no true BM25. BM25 document `i` is chunk `chunk_ids[i]`; `contract_ids[i]` is stored too
so retrieval can restrict results to one contract without a DB round-trip.

Built after chunks are inserted, because the mapping needs their database ids.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
import Stemmer

MAPPING_FILE = "chunk_mapping.json"

_stemmer = Stemmer.Stemmer("english")


def tokenize(texts: list[str]) -> bm25s.tokenization.Tokenized:
    """Lowercase, drop English stopwords, Snowball-stem. Used for both chunks and queries."""
    return bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer, show_progress=False)


@dataclass
class BM25Index:
    retriever: bm25s.BM25
    chunk_ids: list[int]
    contract_ids: list[int]


def build(
    chunk_ids: list[int], contract_ids: list[int], texts: list[str], path: str | Path
) -> BM25Index:
    """Index `texts` (in the given order) and save the index plus id mapping to `path`."""
    if not len(chunk_ids) == len(contract_ids) == len(texts):
        raise ValueError("chunk_ids, contract_ids and texts must be the same length")
    retriever = bm25s.BM25()
    retriever.index(tokenize(texts), show_progress=False)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    retriever.save(path)
    mapping = {"chunk_ids": chunk_ids, "contract_ids": contract_ids}
    (path / MAPPING_FILE).write_text(json.dumps(mapping))
    return BM25Index(retriever, chunk_ids, contract_ids)


def load(path: str | Path) -> BM25Index:
    path = Path(path)
    mapping = json.loads((path / MAPPING_FILE).read_text())
    return BM25Index(bm25s.BM25.load(path), mapping["chunk_ids"], mapping["contract_ids"])
