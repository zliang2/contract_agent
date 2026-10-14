"""Ingestion step 4 (and query-time retrieval): embed text with bge-small-en-v1.5.

Vectors are L2-normalized, so cosine distance in pgvector equals 1 - dot product.
Also exposes the model's tokenizer as a `token_spans` function for the chunker, so chunk
sizes are measured in the same tokens the embedder sees (it truncates at 512).
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from clausewitz.config import get_settings

BATCH_SIZE = 64


@lru_cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embed_model)


def token_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the embedding model's tokens over `text` (no special tokens)."""
    enc = get_model().tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    return [(s, e) for s, e in enc["offset_mapping"] if e > s]


def embed_passages(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """Embed chunk texts. Returns float32 array of shape (n, 384), unit-normalized."""
    return (
        get_model()
        .encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        .astype(np.float32)
    )
