"""Shared fixtures. `db_url` skips DB tests when Postgres isn't reachable."""

import psycopg
import pytest

from clausewitz.config import get_settings


@pytest.fixture(scope="session")
def db_url() -> str:
    url = get_settings().database_url
    try:
        psycopg.connect(url, connect_timeout=2).close()
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d db`")
    return url
