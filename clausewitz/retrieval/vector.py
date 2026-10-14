"""Pipeline stage 2: semantic retrieval with pgvector cosine distance.

Questions get bge's retrieval instruction prefix (chunks are embedded without it, as the
model card recommends for short-query → passage search).

- Whole corpus: the HNSW index (approximate), with `ef_search` raised above k.
- One contract: exact distances over that contract's chunks (1–253 rows). HNSW applies
  the WHERE filter *after* its approximate search, so a filtered HNSW query can return
  fewer than k rows or miss true neighbors; the MATERIALIZED CTE forces the filter first.
"""

from functools import lru_cache

import numpy as np

from clausewitz.db import connect
from clausewitz.ingest.embed import get_model

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1024)
def embed_query(question: str) -> np.ndarray:
    """Unit-normalized query vector. Cached: eval runs ask the same 41 questions repeatedly."""
    return (
        get_model()
        .encode(QUERY_INSTRUCTION + question, normalize_embeddings=True, convert_to_numpy=True)
        .astype(np.float32)
    )


def search(
    question: str, k: int = 50, contract_id: int | None = None, conn=None
) -> list[tuple[int, float]]:
    """Top-k `(chunk_id, cosine_similarity)` for the question."""
    q = embed_query(question)
    own = conn is None
    conn = conn or connect()
    try:
        if contract_id is None:
            with conn.transaction():
                conn.execute(f"SET LOCAL hnsw.ef_search = {max(2 * k, 40)}")
                rows = conn.execute(
                    "SELECT id, 1 - (embedding <=> %s) FROM chunks "
                    "ORDER BY embedding <=> %s LIMIT %s",
                    (q, q, k),
                ).fetchall()
        else:
            rows = conn.execute(
                "WITH c AS MATERIALIZED (SELECT id, embedding FROM chunks WHERE contract_id = %s) "
                "SELECT id, 1 - (embedding <=> %s) FROM c ORDER BY embedding <=> %s, id LIMIT %s",
                (contract_id, q, q, k),
            ).fetchall()
    finally:
        if own:
            conn.close()
    return [(int(i), float(s)) for i, s in rows]
