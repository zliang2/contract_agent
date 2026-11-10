"""Per-request tracing: one structured JSON log line per query (docs/SPEC.md §10).

`trace_record(answer)` turns a pipeline `Answer` into a flat dict with every field the
spec lists: per-stage durations, tokens per model call, generation tokens/sec, cost, and
chunk/claim/refusal counts. `emit(record)` writes it to stdout as one JSON line, which
Cloud Run forwards to Cloud Logging as a structured `jsonPayload` (`severity` and
`message` are the keys Cloud Logging recognizes). Log-based metrics and the dashboard are
built on these fields.
"""

import json
import sys

from clausewitz.pipeline import Answer
from clausewitz.pricing import cost_usd

STAGES = ("bm25", "vector", "fusion", "rerank", "generate", "verify")


def trace_record(a: Answer, prices: dict | None = None) -> dict:
    calls = [
        {
            "stage": stage,
            "model": r.model,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "latency_ms": round(r.latency_ms, 1),
        }
        for stage, results in a.llm_calls.items()
        for r in results
    ]
    costs = [cost_usd(c["model"], c["input_tokens"], c["output_tokens"], prices) for c in calls]
    gen_out = sum(r.output_tokens for r in a.llm_calls.get("generate", []))
    gen_ms = a.timings_ms.get("generate")
    n_claims = len(a.verification.all_claims) if a.verification else 0
    return {
        "severity": "ERROR" if a.error else "INFO",
        "message": "query",
        "trace_id": a.trace_id,
        "duration_ms": {s: round(a.timings_ms[s], 1) for s in STAGES if s in a.timings_ms}
        | {"total": round(a.timings_ms["total"], 1)},
        "llm_calls": calls,
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        # Output tokens per second of generation wall time (includes thinking tokens).
        "generation_tokens_per_s": round(gen_out / (gen_ms / 1000), 1) if gen_ms else None,
        "cost_usd": None if None in costs else round(sum(costs), 6),
        "chunks_retrieved": len(a.retrieval.chunk_ids),
        "claims_generated": n_claims,
        "claims_dropped": a.dropped_claims,
        "refused": a.refused,
        "error": a.error,
    }


def emit(record: dict, stream=None) -> None:
    print(json.dumps(record, separators=(",", ":")), file=stream or sys.stdout, flush=True)
