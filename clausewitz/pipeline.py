"""Query pipeline orchestration (docs/SPEC.md §6).

`answer()` runs all six stages: BM25 → vector → RRF fusion → cross-encoder rerank →
generate → verify, timing each one. `retrieve()` runs only the first four; its `mode`
runs a single strategy so the eval can compare them:

- "bm25":   BM25 top `final_k`
- "vector": vector top `final_k`
- "hybrid": RRF(BM25, vector) top `final_k`
- "rerank": RRF top `rerank_candidates`, reranked, top `final_k` (the production path)
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from clausewitz.config import get_settings
from clausewitz.llm import generate as gen
from clausewitz.llm import verify as ver
from clausewitz.llm.providers import LLMResult, get_provider
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


@dataclass
class Answer:
    """The §6 response, plus internals the eval and tracing need."""

    answer: str
    claims: list[ver.ClaimCheck]
    refused: bool
    dropped_claims: int
    trace_id: str
    retrieval: Retrieval
    verification: ver.Verification | None
    llm_calls: dict[str, list[LLMResult]]
    timings_ms: dict[str, float]
    error: str | None = None

    def to_response(self) -> dict:
        """JSON response for the API (docs/SPEC.md §6)."""
        return {
            "answer": self.answer,
            "claims": [
                {
                    "text": c.text,
                    "citations": [
                        {
                            "chunk_id": x.chunk_id,
                            "contract_id": x.contract_id,
                            "start_char": x.start_char,
                            "end_char": x.end_char,
                            "quote": x.quote,
                        }
                        for x in c.citations
                        if x.quote_found and (x.passed or x.supported is None)
                    ],
                }
                for c in self.claims
            ],
            "refused": self.refused,
            "dropped_claims": self.dropped_claims,
            "trace_id": self.trace_id,
        }


def _load_chunks(conn, chunk_ids: list[int]) -> list[ver.Chunk]:
    rows = conn.execute(
        "SELECT id, contract_id, start_char, text FROM chunks WHERE id = ANY(%s)", (chunk_ids,)
    ).fetchall()
    by_id = {r[0]: ver.Chunk(*r) for r in rows}
    return [by_id[c] for c in chunk_ids]


def answer(
    question: str,
    contract_id: int | None,
    conn,
    verify: ver.Mode = "enforce",
    generate_model: str | None = None,
    verify_model: str | None = None,
) -> Answer:
    """Answer a question with cited claims. `verify` is "enforce", "audit" or "off"."""
    settings = get_settings()
    generate_model = generate_model or settings.generate_model
    verify_model = verify_model or settings.verify_model
    trace_id = uuid.uuid4().hex

    r = retrieve(question, contract_id, conn, mode="rerank")
    timings = dict(r.timings_ms)
    chunks = _load_chunks(conn, r.chunk_ids)

    t = time.perf_counter()
    g = gen.generate(question, chunks, get_provider(generate_model), generate_model)
    timings["generate"] = (time.perf_counter() - t) * 1000
    calls = {"generate": g.attempts, "verify": []}

    if g.answer is None:
        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        return Answer(
            answer=ver.REFUSAL, claims=[], refused=True, dropped_claims=0, trace_id=trace_id,
            retrieval=r, verification=None, llm_calls=calls, timings_ms=timings, error=g.error,
        )  # fmt: skip

    t = time.perf_counter()
    v = ver.verify(
        [c.model_dump() for c in g.answer.claims],
        {c.id: c for c in chunks},
        mode=verify,
        provider=get_provider(verify_model) if verify != "off" else None,
        model=verify_model,
    )
    timings["verify"] = (time.perf_counter() - t) * 1000
    timings["total"] = sum(v_ for k, v_ in timings.items() if k != "total")
    calls["verify"] = v.llm_calls
    return Answer(
        answer=ver.REFUSAL if v.refused else g.answer.answer,
        claims=v.claims,
        refused=v.refused,
        dropped_claims=v.dropped_claims,
        trace_id=trace_id,
        retrieval=r,
        verification=v,
        llm_calls=calls,
        timings_ms=timings,
    )
