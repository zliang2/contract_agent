"""Query pipeline orchestration (docs/SPEC.md §6).

Phase 2 implements the retrieval half: BM25 → vector → RRF fusion → cross-encoder rerank,
with every stage timed. `mode` runs a single strategy so the eval can compare them:

- "bm25":   BM25 top `final_k`
- "vector": vector top `final_k`
- "hybrid": RRF(BM25, vector) top `final_k`
- "rerank": RRF top `rerank_candidates`, reranked, top `final_k` (the production path)
"""

import time
from dataclasses import dataclass, field
from typing import Literal

from clausewitz.retrieval import bm25, fusion, rerank, vector

Mode = Literal["bm25", "vector", "hybrid", "rerank"]

CANDIDATES_K = 50  # per retriever (spec §6 stages 1–2)
RERANK_CANDIDATES = 50  # fused candidates the cross-encoder scores
FINAL_K = 8  # chunks passed to generation (spec §6 stage 4)


@dataclass
class Retrieval:
    chunk_ids: list[int]  # final ranking, best first, length <= final_k
    candidates: dict[str, list[int]] = field(default_factory=dict)  # per-stage rankings
    timings_ms: dict[str, float] = field(default_factory=dict)


def _fetch_texts(conn, chunk_ids: list[int]) -> dict[int, str]:
    rows = conn.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (chunk_ids,)).fetchall()
    return dict(rows)


def retrieve(
    question: str,
    contract_id: int | None,
    conn,
    mode: Mode = "rerank",
    final_k: int = FINAL_K,
) -> Retrieval:
    """Run retrieval for one question. `conn` is an open DB connection (reused across calls)."""
    timings: dict[str, float] = {}
    candidates: dict[str, list[int]] = {}

    def timed(stage, fn, *args, **kwargs):
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        timings[stage] = (time.perf_counter() - t) * 1000
        return out

    bm25_hits = vector_hits = None
    if mode in ("bm25", "hybrid", "rerank"):
        bm25_hits = timed("bm25", bm25.search, question, CANDIDATES_K, contract_id)
        candidates["bm25"] = [c for c, _ in bm25_hits]
    if mode in ("vector", "hybrid", "rerank"):
        vector_hits = timed("vector", vector.search, question, CANDIDATES_K, contract_id, conn)
        candidates["vector"] = [c for c, _ in vector_hits]

    if mode == "bm25":
        final = candidates["bm25"][:final_k]
    elif mode == "vector":
        final = candidates["vector"][:final_k]
    else:
        fused = timed("fusion", fusion.rrf, [bm25_hits, vector_hits])
        candidates["hybrid"] = [c for c, _ in fused]
        if mode == "hybrid":
            final = candidates["hybrid"][:final_k]
        else:
            pool = candidates["hybrid"][:RERANK_CANDIDATES]

            def _rerank():
                texts = _fetch_texts(conn, pool)
                return rerank.rerank(question, [(c, texts[c]) for c in pool], top_n=final_k)

            final = [c for c, _ in timed("rerank", _rerank)]

    timings["total"] = sum(timings.values())
    return Retrieval(chunk_ids=final, candidates=candidates, timings_ms=timings)
