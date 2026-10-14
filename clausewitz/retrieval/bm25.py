"""Pipeline stage 1: keyword retrieval with the BM25 index built at ingestion.

Scores every chunk, then (optionally) keeps only one contract's chunks before taking the
top k. Scoring the full index keeps IDF statistics corpus-wide, and filtering before the
top-k cut means a contract filter never returns fewer results than the contract has.
"""

from functools import lru_cache

import numpy as np

from clausewitz.config import get_settings
from clausewitz.ingest import bm25_index


@lru_cache
def get_index() -> bm25_index.BM25Index:
    return bm25_index.load(get_settings().bm25_index_path)


@lru_cache
def _contract_ids() -> np.ndarray:
    return np.asarray(get_index().contract_ids)


def search(question: str, k: int = 50, contract_id: int | None = None) -> list[tuple[int, float]]:
    """Top-k `(chunk_id, bm25_score)` for the question. Chunks scoring 0 are dropped."""
    idx = get_index()
    tokens = bm25_index.tokenize_query(question)
    if not tokens:
        return []
    scores = idx.retriever.get_scores(tokens)
    candidates = np.arange(len(scores))
    if contract_id is not None:
        candidates = np.flatnonzero(_contract_ids() == contract_id)
    candidates = candidates[scores[candidates] > 0]
    top = candidates[np.argsort(-scores[candidates], kind="stable")[:k]]
    return [(idx.chunk_ids[i], float(scores[i])) for i in top]
