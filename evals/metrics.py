"""Eval metrics (docs/SPEC.md §7). Pure functions over spans; no I/O.

Spans are half-open character ranges `(start, end)` within one contract.
"""

from math import comb

import numpy as np


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if half-open spans share at least one character."""
    return a[0] < b[1] and b[0] < a[1]


def is_relevant(chunk: tuple[int, int], gold: list[tuple[int, int]]) -> bool:
    return any(overlaps(chunk, g) for g in gold)


def hit_at_k(ranked: list[tuple[int, int]], gold: list[tuple[int, int]], k: int) -> bool:
    """Recall@k for one case: does any of the top-k chunk spans overlap a gold span?"""
    return any(is_relevant(span, gold) for span in ranked[:k])


def random_hit_probability(n_chunks: int, n_relevant: int, k: int) -> float:
    """Exact chance that k chunks drawn uniformly without replacement include a relevant one.

    1 - C(n - r, k) / C(n, k). The baseline any retriever must beat within one contract;
    it is 1.0 whenever the contract has <= k chunks and at least one is relevant.
    """
    if n_relevant == 0:
        return 0.0
    if k >= n_chunks:
        return 1.0
    return 1.0 - comb(n_chunks - n_relevant, k) / comb(n_chunks, k)


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": float("nan"), "p95": float("nan")}
    p50, p95 = np.percentile(values, [50, 95])
    return {"p50": float(p50), "p95": float(p95)}


def first_relevant_rank(ranked: list[tuple[int, int]], gold: list[tuple[int, int]]) -> int | None:
    """1-based rank of the first chunk overlapping a gold span, or None if none does.

    hit_at_k(ranked, gold, k) == (rank is not None and rank <= k), so storing this one
    number per case lets recall@k for any k be recomputed later (e.g. after merging shards).
    """
    for rank, span in enumerate(ranked, start=1):
        if is_relevant(span, gold):
            return rank
    return None
