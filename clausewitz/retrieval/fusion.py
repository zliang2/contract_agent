"""Pipeline stage 3: Reciprocal Rank Fusion of the BM25 and vector result lists.

score(chunk) = Σ over lists of 1 / (k + rank), rank starting at 1; a chunk missing from a
list contributes nothing for it. RRF uses only ranks, so BM25 scores and cosine
similarities (different scales) never need normalizing. k = 60 per Cormack et al. (2009).
"""

RRF_K = 60


def rrf(ranked_lists: list[list[tuple[int, float]]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Fuse ranked `(chunk_id, score)` lists. Returns `(chunk_id, rrf_score)`, best first.

    Ties break by chunk id so results are deterministic.
    """
    fused: dict[int, float] = {}
    for results in ranked_lists:
        for rank, (chunk_id, _) in enumerate(results, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
