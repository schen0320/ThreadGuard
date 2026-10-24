"""Evaluation counts abstentions and failures instead of hiding them."""
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .core import Post, run, POLICY_VERSION, POLICY


def load_cases(path, split):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids, texts = set(), set()
    for row in rows:
        if row["id"] in ids or row["target"] in texts:
            raise ValueError("duplicate case id or target across splits")
        ids.add(row["id"]); texts.add(row["target"])
        if row["gold"] not in ("allow", "flag", "review") or row["split"] not in ("dev", "test"):
            raise ValueError("invalid case label or split")
        Post(row["target"], row.get("context", ""))
    selected = [r for r in rows if r["split"] == split]
    if not selected:
        raise ValueError("empty evaluation split")
    return selected


def metrics(rows):
    n = len(rows)
    ratio = lambda a, b: round(a / b, 4) if b else None
    known = [r for r in rows if r["gold"] != "review"]
    answered = [r for r in known if r["prediction"] != "review"]
    tp = sum(r["gold"] == r["prediction"] == "flag" for r in rows)
    fp = sum(r["gold"] == "allow" and r["prediction"] == "flag" for r in rows)
    predicted_flags = sum(r["prediction"] == "flag" for r in known)
    positives = sum(r["gold"] == "flag" for r in rows)
    negatives = sum(r["gold"] == "allow" for r in rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    return {
        "n": n, "accuracy_including_review": ratio(sum(r["gold"] == r["prediction"] for r in rows), n),
        "flag_precision_known_labels": ratio(tp, predicted_flags), "flag_recall": ratio(tp, positives),
        "false_positive_rate": ratio(fp, negatives),
        "flag_or_review_capture_on_violations": ratio(sum(r["gold"] == "flag" and r["prediction"] != "allow" for r in rows), positives),
        "review_rate": ratio(sum(r["prediction"] == "review" for r in rows), n),
        "decisive_coverage_known_labels": ratio(len(answered), len(known)),
        "selective_accuracy_known_labels": ratio(sum(r["gold"] == r["prediction"] for r in answered), len(answered)),
        "provider_errors": sum(r["status"] != "ok" for r in rows),
        "confusion_matrix": {g: {p: sum(r["gold"] == g and r["prediction"] == p for r in rows) for p in ("allow", "flag", "review")} for g in ("allow", "flag", "review")},
        "p50_latency_ms": latencies[(n - 1) // 2] if n else None,
        "p95_latency_ms": latencies[max(0, math.ceil(n * .95) - 1)] if n else None,
    }


def evaluate(path, split, provider, mode, stop_on_error=False, case_ids=None):
    cases = load_cases(path, split)
    split_size = len(cases)
    if case_ids is not None:
        if (not case_ids or len(set(case_ids)) != len(case_ids)
                or not set(case_ids) <= {c["id"] for c in cases}):
            raise ValueError("case_ids must be distinct IDs from the selected split")
        cases = [c for c in cases if c["id"] in set(case_ids)]
    rows, tokens = [], Counter()
    for case in cases:
        # Only target/context reach the provider. Gold, IDs and tags remain evaluator-side.
        result = run(Post(case["target"], case.get("context", "")), provider, mode)
        rows.append({"id": case["id"], "gold": case["gold"], "prediction": result["decision"]["label"],
                     "category": result["decision"]["category"], "status": result["status"],
                     "latency_ms": result["latency_ms"], "tags": case["tags"],
                     "input_sha256": result["input_sha256"], "reason": result["decision"]["reason"]})
        for trace in result["traces"]:
            tokens.update(trace["usage"])
        if stop_on_error and result["status"] != "ok":
            break
    return {"created_at": datetime.now(timezone.utc).isoformat(), "provider": provider.name, "model": provider.model,
            "mode": mode, "split": split, "policy_version": POLICY_VERSION,
            "policy_sha256": hashlib.sha256(POLICY.encode()).hexdigest(),
            "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "synthetic_data": True, "live_llm_attempted": provider.name in ("openai-live", "gemini-live"),
            "live_llm_evaluation": provider.name in ("openai-live", "gemini-live") and any(r["status"] == "ok" for r in rows),
            "available_split_cases": split_size, "selected_case_ids": [c["id"] for c in cases],
            "expected_cases": len(cases), "completed_cases": len(rows),
            "complete": len(rows) == len(cases) and all(r["status"] == "ok" for r in rows),
            "warning": "Small AI-assisted synthetic set, not a blind or production benchmark. Review labels with a human. Same-model passes have correlated errors.",
            "metrics": metrics(rows), "tokens": dict(tokens),
            "subgroups": {tag: metrics([r for r in rows if tag in r["tags"]]) for tag in sorted({tag for r in rows for tag in r["tags"]})},
            "cases": rows}
