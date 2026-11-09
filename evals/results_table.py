"""Print Markdown tables for docs/results.md straight from eval report files.

    uv run python -m evals.results_table retrieval REPORT [--compare OTHER_REPORT]

Numbers in the docs are pasted from this output, never typed by hand (CLAUDE.md).
"""

import argparse
import json
from pathlib import Path

MODES = {
    "bm25": "BM25 only",
    "vector": "Vector only",
    "hybrid": "Hybrid (RRF)",
    "rerank": "Hybrid + rerank",
}


def f(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def retrieval_table(report: dict) -> str:
    m = report["metrics"]
    big = m["contracts_gt_8_chunks"]
    lines = [
        "| Mode | Recall@1 | Recall@3 | Recall@8 | Recall@8, contracts > 8 chunks "
        "| Recall@50 (candidate pool) |",
        "|---|---|---|---|---|---|",
    ]
    for k, name in MODES.items():
        r = m["recall_at_k"][k]
        pool = m["recall@50_candidate_pool"].get(k)
        lines.append(
            f"| {name} | {f(r['1'])} | {f(r['3'])} | {f(r['8'])} | {f(big['recall@8'][k])} "
            f"| {f(pool)} |"
        )
    lines.append(
        f"| Random 8 chunks (exact expectation) | — | — | {f(m['random_baseline_recall@8'])} "
        f"| {f(big['random_baseline_recall@8'])} | — |"
    )
    lat = m["latency_ms"]
    stages = ("bm25", "vector", "fusion", "rerank", "total")
    lines += [
        "",
        f"Cases: {m['n_cases']} answerable {report['config']['split']} cases "
        f"({m['n_cases_contract_gt_8_chunks']} in contracts with more than 8 chunks), out of "
        f"{report['cases']['total']} (contract, clause type) pairs.",
        "",
        f"Per-stage latency on {report['environment']['device']} "
        f"(`{report['environment']['platform']}`), ms p50 / p95: "
        + ", ".join(f"{s} {lat[s]['p50']:.1f} / {lat[s]['p95']:.1f}" for s in stages)
        + ".",
    ]
    return "\n".join(lines)


def compare_recall8(a: dict, b: dict) -> str:
    def label(r: dict) -> str:
        return f"{r['config'].get('question_style', 'cuad_template')} (`{r['git']['commit'][:7]}`)"

    lines = [f"| Mode | {label(a)} | {label(b)} |", "|---|---|---|"]
    for k, name in MODES.items():
        lines.append(
            f"| {name} | {f(a['metrics']['recall@8'][k])} | {f(b['metrics']['recall@8'][k])} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=["retrieval"])
    ap.add_argument("report")
    ap.add_argument("--compare", help="second report: print a recall@8 comparison")
    args = ap.parse_args()
    report = json.loads(Path(args.report).read_text())
    print(retrieval_table(report))
    if args.compare:
        print()
        print(compare_recall8(json.loads(Path(args.compare).read_text()), report))


if __name__ == "__main__":
    main()
