"""Runtime settings, read from environment variables (and `.env` locally).

Every other module gets configuration from `get_settings()`, so model IDs, paths and
credentials are never hard-coded (docs/SPEC.md §2).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://clausewitz:clausewitz@localhost:5432/clausewitz"
    anthropic_api_key: str = ""
    google_cloud_project: str = ""
    generate_model: str = ""
    verify_model: str = ""
    bm25_index_path: str = "indexes/bm25"
    price_table_path: str = "evals/prices.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
