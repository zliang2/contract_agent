"""Query-time retrieval: BM25, vector search, RRF fusion and cross-encoder reranking
(docs/SPEC.md §6 stages 1–4). Each stage returns `(chunk_id, score)` pairs, best first."""
