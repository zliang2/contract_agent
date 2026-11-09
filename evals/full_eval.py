"""Full-pipeline eval: generation + verification + independent judge (docs/SPEC.md §7).

Each case is answered once with the verifier in "audit" mode, which runs every check but
drops nothing. The verifier-OFF condition shows all generated claims; the verifier-ON
condition shows only claims that passed, exactly what "enforce" mode would show. So both
conditions are measured on the *same* generations (a paired comparison) for one
generation's cost.

Metric definitions (per condition; "answered" = not refused):
- ungrounded_rate_judge: share of answered cases with >= 1 shown claim the independent
  judge marks unsupported (evals/judge.py). The headline groundedness metric.
- ungrounded_rate_verifier: the spec's automatic version (>= 1 shown claim failing the
  verifier). Zero by construction when the verifier is on; reported for completeness.
- citation_precision: share of shown citations with a located span that overlaps a gold
  span. Citations whose quote wasn't found have no span and are counted separately.
- refusal_accuracy: on should-refuse cases (no gold span), share refused.
- answer_rate: on answerable cases, share answered (guards against "refuse everything").
- accuracy: on answerable cases, share answered with >= 1 shown citation overlapping gold.
- cost_per_query_usd: generation (+ verification when on) at `PRICE_TABLE_PATH` prices.
  Judge calls are eval-only and reported separately.
"""

import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from clausewitz import pipeline
from clausewitz.llm.providers import get_provider
from clausewitz.pricing import cost_usd
from evals.judge import judge_claim
from evals.metrics import is_relevant, percentiles

CONDITIONS = ("off", "on")


def _call(stage: str, r) -> dict:
    return {
        "stage": stage,
        "model": r.model,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "latency_ms": round(r.latency_ms, 1),
        "refused": r.refused,
    }


def case_key(contract_id: int, clause_type: str) -> str:
    return f"{contract_id}|{clause_type}"


def load_checkpoint(path: Path, run_id: dict) -> dict[str, dict]:
    """Records already completed by an earlier run with the same `run_id`.

    The first line of the checkpoint is the run id (commit + config). A checkpoint from a
    different commit or config is refused, so a report never mixes code versions.
    """
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    if not lines or json.loads(lines[0]) != run_id:
        raise SystemExit(f"{path} is from a different commit/config; delete it to start over")
    records = [json.loads(line) for line in lines[1:] if line.strip()]
    return {case_key(r["contract_id"], r["clause_type"]): r for r in records}


def run_full_cases(
    conn, cases, generate_model: str, verify_model: str, judge_model: str, verify: bool = True,
    checkpoint: Path | None = None, run_id: dict | None = None,
):  # fmt: skip
    """Answer each case and return one record per case. `verify=False` skips verifier calls
    (only the verifier-off condition is then meaningful).

    With `checkpoint`, each finished record is appended to that JSONL file and cases found
    there are skipped, so a run stopped by a daily quota resumes where it left off.
    Raises `QuotaExhausted` (after saving progress) when the provider's daily quota runs out.
    """
    judge = get_provider(judge_model)
    done = load_checkpoint(checkpoint, run_id) if checkpoint else {}
    if checkpoint and not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps(run_id) + "\n")
    records = []
    todo = [c for c in cases if case_key(c.contract_id, c.clause_type) not in done]
    if done:
        print(f"resuming: {len(cases) - len(todo)} of {len(cases)} cases already done")
    for case in tqdm(todo, desc="full eval", mininterval=10):
        a = pipeline.answer(
            case.question, case.contract_id, conn, verify="audit" if verify else "off",
            generate_model=generate_model, verify_model=verify_model,
        )  # fmt: skip
        calls = [_call("generate", r) for r in a.llm_calls["generate"]]
        calls += [_call("verify", r) for r in a.llm_calls["verify"]]
        chunk_text = dict(
            conn.execute(
                "SELECT id, text FROM chunks WHERE id = ANY(%s)", (a.retrieval.chunk_ids,)
            ).fetchall()
        )
        claims = []
        for c in a.verification.all_claims if a.verification else []:
            cited = [chunk_text[x.chunk_id] for x in c.citations if x.chunk_id in chunk_text]
            supported, reason, jcall = judge_claim(c.text, cited, judge, judge_model)
            if jcall is not None:
                calls.append(_call("judge", jcall))
            claims.append(
                {
                    "text": c.text,
                    "verifier_passed": c.passed,
                    "judge_supported": supported,
                    "judge_reason": reason,
                    "citations": [
                        {
                            "chunk_id": x.chunk_id,
                            "quote": x.quote,
                            "start_char": x.start_char,
                            "end_char": x.end_char,
                            "quote_found": x.quote_found,
                            "supported": x.supported,
                            "passed": x.passed,
                            "overlaps_gold": x.quote_found
                            and is_relevant((x.start_char, x.end_char), case.gold),
                        }
                        for x in c.citations
                    ],
                }
            )
        records.append(
            {
                "contract_id": case.contract_id,
                "clause_type": case.clause_type,
                "answerable": case.answerable,
                "error": a.error,
                "answer": a.answer if a.verification else None,
                "claims": claims,
                "calls": calls,
                "timings_ms": {k: round(v, 1) for k, v in a.timings_ms.items()},
            }
        )
        if checkpoint:
            with checkpoint.open("a") as f:
                f.write(json.dumps(records[-1]) + "\n")
    by_key = {**done, **{case_key(r["contract_id"], r["clause_type"]): r for r in records}}
    return [by_key[case_key(c.contract_id, c.clause_type)] for c in cases]


def _shown(record: dict, condition: str) -> list[dict]:
    if condition == "off":
        return record["claims"]
    return [c for c in record["claims"] if c["verifier_passed"]]


def _cost(calls: list[dict], stages: set[str], prices: dict) -> float | None:
    total = 0.0
    for c in calls:
        if c["stage"] not in stages:
            continue
        usd = cost_usd(c["model"], c["input_tokens"], c["output_tokens"], prices)
        if usd is None:
            return None
        total += usd
    return total


def full_metrics(records: list[dict], prices: dict, conditions=CONDITIONS) -> dict:
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else None

    answerable = [r for r in records if r["answerable"]]
    refuse = [r for r in records if not r["answerable"]]
    out: dict = {
        "n_cases": len(records),
        "n_answerable": len(answerable),
        "n_should_refuse": len(refuse),
        "generation_errors": sum(r["error"] is not None for r in records),
        "judge_failures": sum(c["judge_supported"] is None for r in records for c in r["claims"]),
        "claims_generated": sum(len(r["claims"]) for r in records),
        "quote_not_found_rate": mean(
            not x["quote_found"] for r in records for c in r["claims"] for x in c["citations"]
        ),
        "conditions": {},
    }
    for cond in conditions:
        shown = {id(r): _shown(r, cond) for r in records}
        answered = [r for r in records if shown[id(r)]]
        cits = [
            x
            for r in records
            for c in shown[id(r)]
            for x in c["citations"]
            if x["quote_found"] and (cond == "off" or x["passed"])
        ]
        stages = {"generate"} | ({"verify"} if cond == "on" else set())
        costs = [_cost(r["calls"], stages, prices) for r in records]
        latency_keys = ("bm25", "vector", "fusion", "rerank", "generate") + (
            ("verify",) if cond == "on" else ()
        )
        totals = [sum(r["timings_ms"].get(k, 0.0) for k in latency_keys) for r in records]
        out["conditions"][cond] = {
            "n_answered": len(answered),
            "ungrounded_rate_judge": mean(
                any(c["judge_supported"] is False for c in shown[id(r)]) for r in answered
            ),
            "ungrounded_rate_verifier": mean(
                any(not c["verifier_passed"] for c in shown[id(r)]) for r in answered
            ),
            "citation_precision": mean(x["overlaps_gold"] for x in cits),
            "n_citations": len(cits),
            "refusal_accuracy": mean(not shown[id(r)] for r in refuse),
            "answer_rate": mean(bool(shown[id(r)]) for r in answerable),
            "accuracy": mean(
                any(
                    x["overlaps_gold"]
                    for c in shown[id(r)]
                    for x in c["citations"]
                    if cond == "off" or x["passed"]
                )
                for r in answerable
            ),  # fmt: skip
            "dropped_claims_per_answer": mean(len(r["claims"]) - len(shown[id(r)]) for r in records)
            if cond == "on"
            else 0.0,
            "cost_per_query_usd": None if None in costs else mean(costs),
            "latency_ms": percentiles(totals),
        }
    tokens = defaultdict(lambda: {"input": 0, "output": 0, "calls": 0})
    for r in records:
        for c in r["calls"]:
            t = tokens[f"{c['stage']}:{c['model']}"]
            t["input"] += c["input_tokens"]
            t["output"] += c["output_tokens"]
            t["calls"] += 1
    out["tokens"] = dict(tokens)
    judge_costs = [_cost(r["calls"], {"judge"}, prices) for r in records]
    out["judge_cost_total_usd"] = None if None in judge_costs else sum(judge_costs)
    stage_lat = defaultdict(list)
    for r in records:
        for k, v in r["timings_ms"].items():
            stage_lat[k].append(v)
    out["latency_ms_by_stage"] = {k: percentiles(v) for k, v in stage_lat.items()}
    return out
