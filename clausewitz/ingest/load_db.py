"""Ingestion entry point: CUAD -> split -> chunks -> embeddings -> Postgres -> BM25 index.

Run: `uv run python -m clausewitz.ingest.load_db`

The job is a full rebuild: it truncates the tables (resetting ids) and reloads everything,
writing one transaction per contract, then builds the BM25 index from the inserted rows.
Because inputs are sorted and the split is seeded, reruns produce the same ids.
"""

import statistics
import time

from tqdm import tqdm

from clausewitz.config import get_settings
from clausewitz.db import connect, init_schema
from clausewitz.ingest import bm25_index
from clausewitz.ingest.chunk import chunk_contract
from clausewitz.ingest.embed import embed_passages, token_spans
from clausewitz.ingest.load_cuad import load_cuad


def main() -> None:
    t0 = time.perf_counter()
    settings = get_settings()
    contracts, questions = load_cuad()
    print(f"loaded {len(contracts)} contracts, {len(questions)} clause types")

    chunks_per_contract = [
        chunk_contract(c.full_text, token_spans) for c in tqdm(contracts, desc="chunking")
    ]
    texts = [
        c.full_text[ch.start_char : ch.end_char]
        for c, chs in zip(contracts, chunks_per_contract, strict=True)
        for ch in chs
    ]
    vectors = embed_passages(texts, show_progress=True)

    init_schema()
    with connect() as conn:
        conn.execute("TRUNCATE annotations, chunks, contracts RESTART IDENTITY")
        conn.commit()
        v = 0
        for c, chs in tqdm(
            zip(contracts, chunks_per_contract, strict=True), total=len(contracts), desc="loading"
        ):
            with conn.transaction():
                (cid,) = conn.execute(
                    "INSERT INTO contracts (title, split, full_text) VALUES (%s, %s, %s) "
                    "RETURNING id",
                    (c.title, c.split, c.full_text),
                ).fetchone()
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (contract_id, section_path, start_char, end_char, "
                        "text, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                        [
                            (
                                cid,
                                ch.section_path,
                                ch.start_char,
                                ch.end_char,
                                texts[v + i],
                                vectors[v + i],
                            )
                            for i, ch in enumerate(chs)
                        ],
                    )
                    cur.executemany(
                        "INSERT INTO annotations (contract_id, clause_type, start_char, end_char) "
                        "VALUES (%s, %s, %s, %s)",
                        [(cid, g.clause_type, g.start_char, g.end_char) for g in c.gold],
                    )
            v += len(chs)

        rows = conn.execute("SELECT id, contract_id, text FROM chunks ORDER BY id").fetchall()
        n_ann = conn.execute("SELECT count(*) FROM annotations").fetchone()[0]

    bm25_index.build(
        [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], settings.bm25_index_path
    )

    n_test = sum(c.split == "test" for c in contracts)
    lengths = [len(t) for t in texts]
    print(f"contracts: {len(contracts)} ({len(contracts) - n_test} dev / {n_test} test)")
    print(f"chunks: {len(rows)} | avg chunk length: {statistics.mean(lengths):.0f} chars")
    print(f"annotations: {n_ann}")
    print(f"BM25 index: {settings.bm25_index_path} ({len(rows)} docs)")
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
