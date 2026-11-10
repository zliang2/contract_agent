import pytest

from clausewitz import gcs
from clausewitz.config import get_settings


def test_parse_gs_uri():
    assert gcs.parse_gs_uri("gs://b/idx/bm25") == ("b", "idx/bm25/")
    assert gcs.parse_gs_uri("gs://b") == ("b", "")
    with pytest.raises(ValueError):
        gcs.parse_gs_uri("s3://b/x")


class Blob:
    def __init__(self, name, data):
        self.name, self.data = name, data

    def download_to_filename(self, path):
        with open(path, "wb") as f:
            f.write(self.data)


class Client:
    def __init__(self, blobs):
        self.blobs = blobs

    def list_blobs(self, bucket, prefix):
        assert (bucket, prefix) == ("b", "idx/")
        return self.blobs


def test_downloads_index_once(tmp_path, monkeypatch):
    dest = tmp_path / "bm25"
    monkeypatch.setenv("BM25_INDEX_PATH", str(dest))
    monkeypatch.setenv("BM25_INDEX_URI", "gs://b/idx")
    get_settings.cache_clear()
    try:
        blobs = [Blob("idx/params.index.json", b"{}"), Blob("idx/sub/a.npy", b"x")]
        assert gcs.ensure_bm25_index(Client(blobs)) == dest
        assert (dest / "params.index.json").read_bytes() == b"{}"
        assert (dest / "sub" / "a.npy").read_bytes() == b"x"
        gcs.ensure_bm25_index(Client([]))  # already present: no download, no error
    finally:
        get_settings.cache_clear()


def test_empty_prefix_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25"))
    monkeypatch.setenv("BM25_INDEX_URI", "gs://b/idx")
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError):
            gcs.ensure_bm25_index(Client([]))
        assert not (tmp_path / "bm25").exists()
    finally:
        get_settings.cache_clear()
