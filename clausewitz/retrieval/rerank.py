"""Pipeline stage 4: rerank fused candidates with a cross-encoder and keep the top n.

The cross-encoder (ms-marco-MiniLM-L-6-v2) reads question and chunk together, which is
more accurate than comparing separate embeddings but too slow to run over a whole
corpus — hence it only sees the fused shortlist. Inputs over 512 tokens are truncated.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 32


@lru_cache
def get_model() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL, max_length=512)


def rerank(
    question: str, candidates: list[tuple[int, str]], top_n: int = 8
) -> list[tuple[int, float]]:
    """Score `(chunk_id, text)` candidates against the question; return the best `top_n`."""
    if not candidates:
        return []
    scores = get_model().predict(
        [(question, text) for _, text in candidates], batch_size=BATCH_SIZE, show_progress_bar=False
    )
    ranked = sorted(
        zip((cid for cid, _ in candidates), scores, strict=True), key=lambda x: (-x[1], x[0])
    )
    return [(cid, float(s)) for cid, s in ranked[:top_n]]
