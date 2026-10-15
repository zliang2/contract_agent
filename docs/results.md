# Results

Every number here is copied from a report in `evals/reports/`, with the command and git commit that produced it. Tables are generated from the report file rather than typed by hand. Values are shown to 3 decimals; exact values are in the report.

## Phase 2 — Retrieval (dev split)

- **Report:** [`evals/reports/2026-10-15_dev_retrieval_448a055_merged8.json`](../evals/reports/2026-10-15_dev_retrieval_448a055_merged8.json)
- **Commit:** `448a05589ed551671b076c74c33eaf44b026d9df` (clean tree)
- **Command:** `python -m evals.run_eval --split dev --stage retrieval --shard I/8` for I = 0..7, merged with `python -m evals.merge_reports`. Run by the `eval-retrieval` GitHub Actions workflow ([run 38786359724](https://github.com/zliang2/contract_agent/actions/runs/38786359724)), on CPU.
- **Setup:** each case retrieves within its own contract. 50 candidates per retriever, RRF k = 60, the reranker scores the top 50 fused candidates, and the final k is 8. A case is a hit if any of the top-k chunks overlaps a lawyer-annotated gold span.

| Mode | Recall@1 | Recall@3 | Recall@8 | Recall@8, contracts > 8 chunks | Recall@50 (candidate pool) |
|---|---|---|---|---|---|
| BM25 only | 0.433 | 0.618 | 0.771 | 0.757 | 0.944 |
| Vector only | 0.293 | 0.476 | 0.686 | 0.660 | 0.969 |
| Hybrid (RRF) | 0.409 | 0.592 | 0.759 | 0.739 | 0.975 |
| Hybrid + rerank | 0.359 | 0.544 | 0.731 | 0.709 | — |
| Random 8 chunks (exact expectation) | — | — | 0.418 | 0.370 | — |

Cases: 5325 answerable dev cases (4915 in contracts with more than 8 chunks), out of 16728 dev (contract, clause type) pairs.

Per-stage latency on GitHub-hosted CPU runners (ms, p50 / p95): bm25 0.5 / 0.8, vector 0.8 / 23.9, fusion 0.0 / 0.1, rerank 1978.2 / 3894.9, total 1981.3 / 3898.4.

### What the numbers show

- Every mode beats the random-8 baseline by a wide margin (recall@8 0.686–0.771 vs 0.418).
- **BM25 alone has the highest recall@1, @3 and @8.** RRF fusion with vector search does not beat BM25 at the top ranks here, though it gives the best candidate pool (recall@50 0.975).
- **The cross-encoder reranker lowers recall** compared with the fused list it reorders: recall@8 drops from 0.759 to 0.731, and recall@1 from 0.409 to 0.359. The right chunk is almost always in the pool it reranks (0.975), so the loss comes from the reranker's ordering, not from retrieval misses.
- Latency is dominated by the reranker on CPU runners (p50 about 2 s); the other stages take milliseconds.

### Hypotheses (not yet tested)

These explain the reranker result but have **not** been measured:

1. **Query wording.** CUAD questions are long templates ("Highlight the parts (if any) of this contract related to "X" that should be reviewed by a lawyer. Details: …"). The ms-marco cross-encoder was trained on short web search queries, and the shared boilerplate may swamp the clause-specific words. BM25 is less affected: stopword removal and IDF down-weight terms that appear in every question.
2. **Domain mismatch.** The reranker was trained on web passages, not contract language.
3. **Truncation.** Question plus chunk can exceed the cross-encoder's 512-token input, cutting off the end of long chunks.

Testing these on the dev split (for example, removing the template boilerplate from the query, or trying a different reranker) is a Phase 2 follow-up for review. The test split has not been used.
