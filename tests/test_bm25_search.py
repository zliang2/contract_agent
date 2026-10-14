"""BM25 search over a small temporary index (no DB or downloaded data needed)."""

import pytest

from clausewitz.ingest import bm25_index
from clausewitz.retrieval import bm25

TEXTS = [
    "Either party may terminate this Agreement for convenience.",  # chunk 10, contract 1
    "This Agreement is governed by the laws of Delaware.",  # chunk 11, contract 1
    "Either party may terminate upon material breach.",  # chunk 20, contract 2
    "Royalties are payable quarterly.",  # chunk 21, contract 2
]


@pytest.fixture
def tiny_index(tmp_path, monkeypatch):
    idx = bm25_index.build([10, 11, 20, 21], [1, 1, 2, 2], TEXTS, tmp_path)
    monkeypatch.setattr(bm25, "get_index", lambda: idx)
    bm25._contract_ids.cache_clear()
    yield idx
    bm25._contract_ids.cache_clear()


def test_contract_filter_applies_before_top_k(tiny_index):
    assert [c for c, _ in bm25.search("terminate", k=1, contract_id=2)] == [20]
    assert {c for c, _ in bm25.search("terminate", k=5)} == {10, 20}


def test_zero_score_and_stopword_queries_return_nothing(tiny_index):
    assert bm25.search("royalties", contract_id=1) == []
    assert bm25.search("the of and") == []
    assert bm25.search("zzqx") == []


def test_results_are_sorted_by_score(tiny_index):
    scores = [s for _, s in bm25.search("either party terminate agreement")]
    assert scores == sorted(scores, reverse=True)
