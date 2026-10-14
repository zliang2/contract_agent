"""Phase 0 acceptance: the vector extension and schema apply against real Postgres."""

import numpy as np
import pytest

from clausewitz.db import connect, init_schema

pytestmark = pytest.mark.db


def test_schema_applies_and_is_idempotent(db_url):
    init_schema(db_url)
    init_schema(db_url)  # second run must not fail
    with connect(db_url) as conn:
        ext = conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()
        assert ext is not None
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
        }
        assert {"contracts", "chunks", "annotations"} <= tables


def test_vector_roundtrip(db_url):
    init_schema(db_url)
    with connect(db_url) as conn:
        v = np.arange(384, dtype=np.float32) / 384
        out = conn.execute("SELECT %s::vector(384)", (v,)).fetchone()[0]
        assert np.allclose(out.to_numpy(), v)  # adapter returns a pgvector.Vector
        conn.rollback()
