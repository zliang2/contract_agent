import pytest

from clausewitz.retrieval.fusion import RRF_K, rrf


def test_rrf_scores_are_sum_of_reciprocal_ranks():
    fused = dict(rrf([[(1, 9.0), (2, 5.0)], [(2, 0.9), (3, 0.8)]]))
    assert fused[1] == pytest.approx(1 / (RRF_K + 1))
    assert fused[2] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert fused[3] == pytest.approx(1 / (RRF_K + 2))


def test_rrf_rewards_agreement_over_one_top_rank():
    # chunk 2 is 2nd in both lists; chunks 1 and 3 are 1st in one list only.
    fused = rrf([[(1, 0), (2, 0)], [(3, 0), (2, 0)]])
    assert fused[0][0] == 2


def test_rrf_ignores_raw_scores():
    a = rrf([[(1, 1000.0), (2, 0.001)], [(2, 0.5), (1, 0.4)]])
    b = rrf([[(1, 1.0), (2, 0.9)], [(2, 0.5), (1, 0.4)]])
    assert a == b


def test_rrf_ties_break_by_chunk_id_and_handles_empty():
    assert [c for c, _ in rrf([[(7, 0)], [(3, 0)]])] == [3, 7]
    assert rrf([[], []]) == []
    assert rrf([[(5, 1.0)], []]) == [(5, pytest.approx(1 / (RRF_K + 1)))]
