"""Postgres access: opening connections and applying `schema.sql`.

Used by ingestion (writes), retrieval (vector search) and the API (reads).
"""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from clausewitz.config import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(url: str | None = None, *, vector: bool = True) -> psycopg.Connection:
    """Open a connection. With `vector=True`, numpy arrays map to the pgvector type.

    Registering the vector type requires the extension to exist, so `init_schema`
    connects with `vector=False`.
    """
    conn = psycopg.connect(url or get_settings().database_url)
    if vector:
        register_vector(conn)
    return conn


def init_schema(url: str | None = None) -> None:
    """Create the extension, tables and indexes if they don't exist."""
    with connect(url, vector=False) as conn:
        conn.execute(SCHEMA_PATH.read_text())


if __name__ == "__main__":
    init_schema()
    print("schema applied")
