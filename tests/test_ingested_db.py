"""Checks on the loaded database (Phase 1 acceptance). Skipped until ingestion has run."""

from pathlib import Path

import pytest

from clausewitz.config import get_settings
from clausewitz.db import connect

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def conn(db_url):
    # Plain SQL only; vector=False so this skips cleanly on a fresh DB without the extension.
    with connect(db_url, vector=False) as c:
        exists = c.execute("SELECT to_regclass('chunks') IS NOT NULL").fetchone()[0]
        if not exists or not c.execute("SELECT EXISTS (SELECT 1 FROM chunks)").fetchone()[0]:
            pytest.skip("database not ingested; run `uv run python -m clausewitz.ingest.load_db`")
        yield c


def test_every_chunk_matches_contract_text(conn):
    # Postgres substr is 1-based and counts characters, like Python str slicing.
    bad = conn.execute(
        """SELECT count(*) FROM chunks ch JOIN contracts c ON c.id = ch.contract_id
           WHERE substr(c.full_text, ch.start_char + 1, ch.end_char - ch.start_char) <> ch.text"""
    ).fetchone()[0]
    assert bad == 0


def test_contracts_and_split(conn):
    rows = dict(conn.execute("SELECT split, count(*) FROM contracts GROUP BY split").fetchall())
    assert rows == {"dev": 408, "test": 102}


def test_annotations_within_contract_bounds(conn):
    n, bad = conn.execute(
        """SELECT count(*), count(*) FILTER (WHERE a.start_char < 0
                  OR a.end_char > length(c.full_text) OR a.start_char >= a.end_char)
           FROM annotations a JOIN contracts c ON c.id = a.contract_id"""
    ).fetchone()
    assert n == 13823 and bad == 0


def test_bm25_mapping_matches_chunks(conn):
    from clausewitz.ingest import bm25_index

    path = Path(get_settings().bm25_index_path)
    if not path.exists():
        pytest.skip("BM25 index not built")
    idx = bm25_index.load(path)
    rows = conn.execute("SELECT id, contract_id FROM chunks ORDER BY id").fetchall()
    assert idx.chunk_ids == [r[0] for r in rows]
    assert idx.contract_ids == [r[1] for r in rows]
