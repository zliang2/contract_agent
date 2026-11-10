"""Fetch the BM25 index from Cloud Storage at API startup (docs/SPEC.md §10).

The index is built at ingestion time, next to the database it maps to (`chunks.id`), so
it ships with the data rather than in the image: re-ingesting means uploading a new
index, not rebuilding the container. Locally `BM25_INDEX_URI` is empty and the index is
read from `BM25_INDEX_PATH` as usual.
"""

from pathlib import Path

from clausewitz.config import get_settings


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """`gs://bucket/some/prefix` -> ("bucket", "some/prefix/")."""
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs://bucket/prefix, got {uri!r}")
    bucket, _, prefix = uri[len("gs://") :].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri!r}")
    return bucket, prefix.strip("/") + "/" if prefix.strip("/") else ""


def ensure_bm25_index(client=None) -> Path:
    """Download the index from `BM25_INDEX_URI` unless `BM25_INDEX_PATH` already exists."""
    settings = get_settings()
    dest = Path(settings.bm25_index_path)
    if dest.exists() or not settings.bm25_index_uri:
        return dest
    if client is None:
        from google.cloud import storage

        client = storage.Client()
    bucket, prefix = parse_gs_uri(settings.bm25_index_uri)
    tmp = dest.with_name(dest.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    n = 0
    for blob in client.list_blobs(bucket, prefix=prefix):
        name = blob.name[len(prefix) :]
        if not name or name.endswith("/"):
            continue
        target = tmp / name
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        n += 1
    if n == 0:
        raise RuntimeError(f"no files under {settings.bm25_index_uri}")
    tmp.rename(dest)  # only a complete download becomes the index
    return dest
