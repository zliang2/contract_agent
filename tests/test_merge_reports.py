import pytest

from evals.merge_reports import merge
from evals.run_eval import retrieval_metrics


def record(rank, n_chunks=20):
    ranks = {m: rank for m in ("bm25", "vector", "hybrid", "rerank")}
    return {
        "contract_id": 1,
        "clause_type": "x",
        "n_chunks": n_chunks,
        "n_relevant": 1,
        "first_relevant_rank": ranks,
        "timings_ms": {"total": 1.0},
    }


def shard(i, n, records, commit="abc"):
    return {
        "command": f"run --shard {i}/{n}",
        "git": {"commit": commit, "dirty": False},
        "date": "2026-10-15T00:00:00+00:00",
        "duration_s": 1.0,
        "environment": {},
        "config": {"split": "dev", "stage": "retrieval", "limit": None, "shard": f"{i}/{n}"},
        "cases": {"total": 10, "answerable_evaluated": len(records)},
        "records": records,
    }


def test_merge_recomputes_metrics_over_all_records():
    a = [record(1), record(None)]
    b = [record(9), record(3, n_chunks=5)]
    merged = merge([shard(1, 2, b), shard(0, 2, a)])
    assert merged["metrics"] == retrieval_metrics(a + b)
    assert merged["metrics"]["recall@8"]["rerank"] == pytest.approx(2 / 4)
    assert merged["metrics"]["recall_at_k"]["rerank"]["1"] == pytest.approx(1 / 4)


def test_merge_rejects_missing_or_mismatched_shards():
    with pytest.raises(ValueError, match="exactly once"):
        merge([shard(0, 3, []), shard(1, 3, [])])
    with pytest.raises(ValueError, match="commits"):
        merge([shard(0, 2, [], "abc"), shard(1, 2, [], "def")])
