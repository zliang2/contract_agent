from math import comb

import pytest

from evals.metrics import hit_at_k, overlaps, percentiles, random_hit_probability


def test_overlaps_is_half_open():
    assert overlaps((0, 10), (9, 20))
    assert not overlaps((0, 10), (10, 20))  # touching, no shared character
    assert overlaps((5, 6), (0, 100))


def test_hit_at_k_only_looks_at_top_k():
    ranked = [(0, 10), (100, 110), (200, 210)]
    gold = [(205, 207)]
    assert not hit_at_k(ranked, gold, 2)
    assert hit_at_k(ranked, gold, 3)
    assert not hit_at_k(ranked, [], 3)


def test_random_hit_probability_matches_formula_and_edges():
    assert random_hit_probability(20, 2, 8) == pytest.approx(1 - comb(18, 8) / comb(20, 8))
    assert random_hit_probability(8, 1, 8) == 1.0  # top-8 of an 8-chunk contract
    assert random_hit_probability(30, 0, 8) == 0.0
    assert random_hit_probability(30, 30, 1) == 1.0


def test_percentiles():
    p = percentiles([float(x) for x in range(1, 101)])
    assert p["p50"] == pytest.approx(50.5) and p["p95"] == pytest.approx(95.05)


def test_first_relevant_rank_agrees_with_hit_at_k():
    from evals.metrics import first_relevant_rank

    ranked = [(0, 10), (100, 110), (200, 210)]
    for gold in ([(205, 207)], [(0, 1)], [(500, 600)], []):
        rank = first_relevant_rank(ranked, gold)
        for k in (1, 2, 3):
            assert hit_at_k(ranked, gold, k) == (rank is not None and rank <= k)
