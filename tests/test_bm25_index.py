from clausewitz.ingest import bm25_index


def test_build_save_load_roundtrip(tmp_path):
    texts = [
        "Either party may terminate this Agreement upon thirty days notice.",
        "This Agreement shall be governed by the laws of Delaware.",
        "Licensee shall pay royalties quarterly.",
    ]
    bm25_index.build([11, 12, 13], [1, 1, 2], texts, tmp_path)
    idx = bm25_index.load(tmp_path)
    assert idx.chunk_ids == [11, 12, 13] and idx.contract_ids == [1, 1, 2]
    docs, _ = idx.retriever.retrieve(bm25_index.tokenize(["governing law"]), k=1)
    assert idx.chunk_ids[int(docs[0, 0])] == 12
    docs, _ = idx.retriever.retrieve(bm25_index.tokenize(["terminated"]), k=1)  # stemming
    assert idx.chunk_ids[int(docs[0, 0])] == 11
