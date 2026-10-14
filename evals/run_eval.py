"""Run an evaluation and write a JSON report to evals/reports/ (docs/SPEC.md §7).

    uv run python -m evals.run_eval --split dev --stage retrieval [--subset FILE] [--limit N]

Retrieval stage: for every answerable case, retrieve within the case's contract and
measure recall@k for BM25-only, vector-only, hybrid (RRF) and hybrid + rerank. The
pipeline runs once per case in "rerank" mode; the other modes' top-k are read from its
per-stage rankings, which equals running them separately because every stage is
deterministic. Also reports an exact random-k baseline and recall restricted to contracts
with more than k chunks (in smaller contracts, top-k is the whole contract).

Every report records the git commit (and whether the tree was dirty), date, config and
metrics. Numbers quoted in docs must come from a report from a clean tree.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from clausewitz import pipeline
from clausewitz.config import get_settings
from clausewitz.db import connect
from clausewitz.retrieval.rerank import RERANK_MODEL
from evals.build_cases import build_cases
from evals.metrics import hit_at_k, is_relevant, percentiles, random_hit_probability

REPORTS_DIR = Path("evals/reports")
KS = (1, 3, 8)
POOL_K = pipeline.CANDIDATES_K
MODES = ("bm25", "vector", "hybrid", "rerank")


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def eval_retrieval(conn, cases, k_final: int) -> dict:
    spans: dict[int, list[tuple[int, int, int]]] = defaultdict(list)  # contract -> chunks
    for cid, chunk_id, s, e in conn.execute(
        "SELECT contract_id, id, start_char, end_char FROM chunks ORDER BY id"
    ):
        spans[cid].append((chunk_id, s, e))
    span_of = {chunk_id: (s, e) for rows in spans.values() for chunk_id, s, e in rows}

    hits = {m: {k: [] for k in KS} for m in MODES}
    pool_hits = {m: [] for m in ("bm25", "vector", "hybrid")}
    random_k = []
    big = []  # case is in a contract with more than k_final chunks
    timings = defaultdict(list)

    for case in tqdm(cases, desc="retrieval eval"):
        r = pipeline.retrieve(case.question, case.contract_id, conn, mode="rerank")
        ranked = {
            "bm25": r.candidates["bm25"],
            "vector": r.candidates["vector"],
            "hybrid": r.candidates["hybrid"],
            "rerank": r.chunk_ids,
        }
        for m in MODES:
            ranked_spans = [span_of[c] for c in ranked[m]]
            for k in KS:
                hits[m][k].append(hit_at_k(ranked_spans, case.gold, k))
        for m in pool_hits:
            pool_hits[m].append(hit_at_k([span_of[c] for c in ranked[m]], case.gold, POOL_K))
        contract_chunks = spans[case.contract_id]
        n_rel = sum(is_relevant((s, e), case.gold) for _, s, e in contract_chunks)
        random_k.append(random_hit_probability(len(contract_chunks), n_rel, k_final))
        big.append(len(contract_chunks) > k_final)
        for stage, ms in r.timings_ms.items():
            timings[stage].append(ms)

    def mean(xs) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def restrict(xs):
        return [x for x, b in zip(xs, big, strict=True) if b]

    return {
        "n_cases": len(cases),
        "n_cases_contract_gt_k": sum(big),
        f"recall@{k_final}": {m: mean(hits[m][k_final]) for m in MODES},
        "recall_at_k": {m: {str(k): mean(hits[m][k]) for k in KS} for m in MODES},
        f"recall@{POOL_K}_candidate_pool": {m: mean(v) for m, v in pool_hits.items()},
        f"random_baseline_recall@{k_final}": mean(random_k),
        f"contracts_gt_{k_final}_chunks": {
            f"recall@{k_final}": {m: mean(restrict(hits[m][k_final])) for m in MODES},
            f"random_baseline_recall@{k_final}": mean(restrict(random_k)),
        },
        "latency_ms": {stage: percentiles(v) for stage, v in timings.items()},
    }


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--stage", choices=["retrieval", "full"], required=True)
    ap.add_argument("--subset", help="JSON list of {title, clause_type}")
    ap.add_argument("--limit", type=int, help="first N answerable cases (smoke runs)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--model")
    args = ap.parse_args(argv)
    if args.stage == "full":
        sys.exit("--stage full arrives in Phase 3")

    started = time.time()
    with connect() as conn:
        cases = build_cases(conn, args.split, args.subset)
        answerable = [c for c in cases if c.answerable]
        if args.limit:
            answerable = answerable[: args.limit]
        metrics = eval_retrieval(conn, answerable, pipeline.FINAL_K)

    git = git_state()
    report = {
        "command": "python -m evals.run_eval " + " ".join(sys.argv[1:] if argv is None else argv),
        "git": git,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 1),
        "config": {
            "split": args.split,
            "stage": args.stage,
            "subset": args.subset,
            "limit": args.limit,
            "scope": "single contract (contract_id given)",
            "candidates_k": pipeline.CANDIDATES_K,
            "rerank_candidates": pipeline.RERANK_CANDIDATES,
            "final_k": pipeline.FINAL_K,
            "embed_model": get_settings().embed_model,
            "rerank_model": RERANK_MODEL,
        },
        "cases": {"total": len(cases), "answerable_evaluated": len(answerable)},
        "metrics": metrics,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{report['date'][:10]}_{args.split}_{args.stage}_{git['commit'][:7]}"
    if args.limit:
        name += f"_limit{args.limit}"
    if git["dirty"]:
        name += "_dirty"
    path = REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    k = pipeline.FINAL_K
    print(
        f"\n{len(answerable)} answerable cases ({metrics['n_cases_contract_gt_k']} in "
        f"contracts with >{k} chunks)"
    )
    print(f"{'mode':<8} {'R@1':>6} {'R@3':>6} {'R@8':>6}  R@8 (contracts >{k} chunks)")
    for m in MODES:
        r = metrics["recall_at_k"][m]
        big = metrics[f"contracts_gt_{k}_chunks"][f"recall@{k}"][m]
        print(f"{m:<8} {r['1']:>6.3f} {r['3']:>6.3f} {r['8']:>6.3f}  {big:.3f}")
    rb = metrics[f"random_baseline_recall@{k}"]
    rb_big = metrics[f"contracts_gt_{k}_chunks"][f"random_baseline_recall@{k}"]
    print(f"{'random':<8} {'':>6} {'':>6} {rb:>6.3f}  {rb_big:.3f}")
    if git["dirty"]:
        print("WARNING: working tree is dirty; don't quote these numbers.")
    print(f"report: {path}")
    return path


if __name__ == "__main__":
    main()
