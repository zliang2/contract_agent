"""Retrieval against the ingested database. Skipped until ingestion has run.

Loads the embedding and reranker models (downloaded on first use)."""

import pytest

from clausewitz import pipeline
from clausewitz.db import connect
from clausewitz.retrieval import vector

pytestmark = pytest.mark.db

QUESTION = "Which state's law governs this agreement?"


@pytest.fixture(scope="module")
def conn(db_url):
    with connect(db_url, vector=False) as c:
        exists = c.execute("SELECT to_regclass('chunks') IS NOT NULL").fetchone()[0]
        if not exists or not c.execute("SELECT EXISTS (SELECT 1 FROM chunks)").fetchone()[0]:
            pytest.skip("database not ingested")
    with connect(db_url) as c:
        yield c


@pytest.fixture(scope="module")
def big_contract(conn):
    """A contract with more chunks than the candidate limit."""
    return conn.execute(
        "SELECT contract_id, count(*) FROM chunks GROUP BY contract_id "
        "HAVING count(*) > %s ORDER BY contract_id LIMIT 1",
        (pipeline.CANDIDATES_K,),
    ).fetchone()


def test_vector_contract_filter_is_exact(conn, big_contract):
    contract_id, _ = big_contract
    hits = vector.search(QUESTION, k=pipeline.CANDIDATES_K, contract_id=contract_id, conn=conn)
    assert len(hits) == pipeline.CANDIDATES_K
    ids = [c for c, _ in hits]
    owners = conn.execute(
        "SELECT DISTINCT contract_id FROM chunks WHERE id = ANY(%s)", (ids,)
    ).fetchall()
    assert owners == [(contract_id,)]
    # Exact search: the top hit equals brute-force best over the contract.
    q = vector.embed_query(QUESTION)
    (best,) = conn.execute(
        "SELECT id FROM chunks WHERE contract_id = %s ORDER BY embedding <=> %s, id LIMIT 1",
        (contract_id, q),
    ).fetchone()
    assert ids[0] == best
    sims = [s for _, s in hits]
    assert sims == sorted(sims, reverse=True)


def test_vector_corpus_search_returns_k(conn):
    assert len(vector.search(QUESTION, k=20, conn=conn)) == 20


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid", "rerank"])
def test_pipeline_modes_return_final_k_from_one_contract(conn, big_contract, mode):
    contract_id, _ = big_contract
    r = pipeline.retrieve(QUESTION, contract_id, conn, mode=mode)
    assert 0 < len(r.chunk_ids) <= pipeline.FINAL_K
    assert len(set(r.chunk_ids)) == len(r.chunk_ids)
    owners = conn.execute(
        "SELECT DISTINCT contract_id FROM chunks WHERE id = ANY(%s)", (r.chunk_ids,)
    ).fetchall()
    assert owners == [(contract_id,)]
    assert r.timings_ms["total"] > 0


def test_rerank_mode_reorders_the_fused_pool(conn, big_contract):
    contract_id, _ = big_contract
    r = pipeline.retrieve(QUESTION, contract_id, conn, mode="rerank")
    pool = r.candidates["hybrid"][: pipeline.RERANK_CANDIDATES]
    assert set(r.chunk_ids) <= set(pool)
