# Results

Every number here is copied from a report in `evals/reports/`, with the command and git commit that produced it. Tables are printed from report files by `python -m evals.results_table`, not typed by hand. Values are shown to 3 decimals; exact values are in the reports.

## Phase 2 — Retrieval (dev split)

Each case retrieves within its own contract. 50 candidates per retriever, RRF k = 60, the reranker scores the top 50 fused candidates, and the final k is 8. A case is a hit if any of the top-k chunks overlaps a lawyer-annotated gold span. Both runs used the `eval-retrieval` GitHub Actions workflow on CPU runners: `python -m evals.run_eval --split dev --stage retrieval --shard I/8` for I = 0..7, merged with `python -m evals.merge_reports`.

### Current setup: plain questions

- **Report:** [`evals/reports/2026-10-15_dev_retrieval_154c41f_merged8.json`](../evals/reports/2026-10-15_dev_retrieval_154c41f_merged8.json)
- **Commit:** `154c41f1c5fecefe78915a3159570f512eb77dc4` (clean tree), [run 38797440896](https://github.com/zliang2/contract_agent/actions/runs/38797440896)
- **Questions:** `--question-style plain` (the default): CUAD's question rewritten as "\<clause type\>: \<details\>" (see `docs/decisions.md`).

| Mode | Recall@1 | Recall@3 | Recall@8 | Recall@8, contracts > 8 chunks | Recall@50 (candidate pool) |
|---|---|---|---|---|---|
| BM25 only | 0.491 | 0.668 | 0.805 | 0.800 | 0.925 |
| Vector only | 0.328 | 0.522 | 0.736 | 0.714 | 0.980 |
| Hybrid (RRF) | 0.446 | 0.640 | 0.811 | 0.795 | 0.984 |
| Hybrid + rerank | 0.482 | 0.698 | 0.856 | 0.844 | — |
| Random 8 chunks (exact expectation) | — | — | 0.418 | 0.370 | — |

Cases: 5325 answerable dev cases (4915 in contracts with more than 8 chunks), out of 16728 (contract, clause type) pairs.

Per-stage latency on cpu (`Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`), ms p50 / p95: bm25 0.5 / 0.7, vector 0.8 / 23.8, fusion 0.0 / 0.1, rerank 2127.7 / 4161.2, total 2132.5 / 4167.6.

| Mode | cuad_template (`448a055`) | plain (`154c41f`) |
|---|---|---|
| BM25 only | 0.771 | 0.805 |
| Vector only | 0.686 | 0.736 |
| Hybrid (RRF) | 0.759 | 0.811 |
| Hybrid + rerank | 0.731 | 0.856 |

### Earlier run: CUAD's verbatim annotation template

- **Report:** [`evals/reports/2026-10-15_dev_retrieval_448a055_merged8.json`](../evals/reports/2026-10-15_dev_retrieval_448a055_merged8.json)
- **Commit:** `448a05589ed551671b076c74c33eaf44b026d9df` (clean tree), [run 38786359724](https://github.com/zliang2/contract_agent/actions/runs/38786359724)
- **Questions:** verbatim CUAD template (this was before `--question-style` existed; equivalent to `cuad_template`).

| Mode | Recall@1 | Recall@3 | Recall@8 | Recall@8, contracts > 8 chunks | Recall@50 (candidate pool) |
|---|---|---|---|---|---|
| BM25 only | 0.433 | 0.618 | 0.771 | 0.757 | 0.944 |
| Vector only | 0.293 | 0.476 | 0.686 | 0.660 | 0.969 |
| Hybrid (RRF) | 0.409 | 0.592 | 0.759 | 0.739 | 0.975 |
| Hybrid + rerank | 0.359 | 0.544 | 0.731 | 0.709 | — |
| Random 8 chunks (exact expectation) | — | — | 0.418 | 0.370 | — |

Cases: 5325 answerable dev cases (4915 in contracts with more than 8 chunks), out of 16728 (contract, clause type) pairs.

Per-stage latency on cpu (`Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`), ms p50 / p95: bm25 0.5 / 0.8, vector 0.8 / 23.9, fusion 0.0 / 0.1, rerank 1978.2 / 3894.9, total 1981.3 / 3898.4.

### What the numbers show

- Every mode beats the random-8 baseline by a wide margin in both runs.
- **With plain questions, hybrid + rerank has the best recall@3 and recall@8** (0.856 at 8, vs 0.811 for hybrid without reranking and 0.805 for BM25). At recall@1, BM25 is slightly higher (0.491 vs 0.482).
- **With the verbatim template, the reranker lowered recall** (recall@8 0.731 vs hybrid 0.759). Rewording the questions raised every mode, and the reranker the most (+0.125 recall@8). This is consistent with the earlier hypothesis that the long template hurt the ms-marco cross-encoder, which was trained on short search queries.
- The fused candidate pool contains a relevant chunk for 0.984 of plain-question cases, so reranking quality, not candidate recall, sets the ceiling.
- Latency is dominated by the reranker on CPU runners (p50 about 2.1 s); the other stages take milliseconds.

### CI gate subset (`ci_50`)

- **Report:** [`evals/reports/2026-11-10_dev_retrieval_5129e94_ci_50.json`](../evals/reports/2026-11-10_dev_retrieval_5129e94_ci_50.json)
- **Commit:** `5129e94e9b335aa9992495bc025e7e365dce603c` (clean tree), run locally (macOS, Apple silicon)
- **Command:** `python -m evals.run_eval --split dev --stage retrieval --subset evals/subsets/ci_50.json`
- **Cases:** 50 answerable dev cases drawn by `python -m evals.make_subset --split dev --n 50 --seed 0` (47 contracts, 28 clause types).

| Mode | Recall@1 | Recall@3 | Recall@8 |
|---|---|---|---|
| BM25 only | 0.520 | 0.700 | 0.800 |
| Vector only | 0.320 | 0.460 | 0.760 |
| Hybrid (RRF) | 0.400 | 0.680 | 0.860 |
| Hybrid + rerank | 0.520 | 0.680 | 0.920 |

This is the baseline the CI gate compares against (`evals/subsets/ci_50_baseline.json`). With only 50 cases, one case is 2 points; these values come from a small sample and are not a replacement for the full dev-split numbers above.

### Not tested

- Whether a contract-domain or larger reranker would do better; whether cross-encoder truncation at 512 tokens costs recall.
- The test split has not been used.
