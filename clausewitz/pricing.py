"""Token cost from a price table (`PRICE_TABLE_PATH`, JSON).

Format: {"models": {"<model id>": {"input_per_mtok": float, "output_per_mtok": float}},
"source": "...", "as_of": "YYYY-MM-DD"}. A model missing from the table has unknown
cost (None), never an estimate.
"""

import json
from functools import lru_cache
from pathlib import Path

from clausewitz.config import get_settings


@lru_cache
def load_prices(path: str | None = None) -> dict:
    p = Path(path or get_settings().price_table_path)
    return json.loads(p.read_text()) if p.exists() else {"models": {}}


def cost_usd(model: str, input_tokens: int, output_tokens: int, prices: dict | None = None):
    """Dollar cost of one call, or None if the model has no listed price."""
    table = (prices or load_prices())["models"]
    price = table.get(model)
    if price is None or price.get("input_per_mtok") is None:
        return None
    return (
        input_tokens * price["input_per_mtok"] + output_tokens * price["output_per_mtok"]
    ) / 1_000_000
