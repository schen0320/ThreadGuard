"""Manual, small-batch Gemini evaluation using a user-confirmed Free Tier project."""
import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .core import Post, run
from .evaluate import evaluate, load_cases
from .live_eval import CappedProvider, save
from .providers import GeminiProvider, KeywordBaseline, ProviderError

DATA_SHA256 = "c147cdbd414ae81d6a8b26d156179f33753440c9ddd9c6d74b5f336ffd9f32bb"
BATCH_SIZE = 5
MAX_CALLS = 17  # Two smoke requests + five single-pass + ten two-pass.


def select_ids(data, batch):
    if type(batch) is not int or batch not in (1, 2, 3, 4):
        raise ValueError("batch must be 1, 2, 3 or 4")
    # The free tier may use inputs to improve products. Reject modified fixtures.
    if hashlib.sha256(Path(data).read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Only the reviewed synthetic fixture file is allowed")
    cases = load_cases(data, "dev")
    if len(cases) != 20:
        raise ValueError("Expected 20 development fixtures")
    start = (batch - 1) * BATCH_SIZE
    return [c["id"] for c in cases[start:start + BATCH_SIZE]]


def execute(data, output, provider, batch=1, *, confirm_free_tier=False):
    if confirm_free_tier is not True:
        raise ValueError("Confirm that the key's project is Free Tier with billing disabled")
    ids = select_ids(data, batch)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    capped = CappedProvider(provider, MAX_CALLS)
    started = datetime.now(timezone.utc).isoformat()
    baseline = evaluate(data, "dev", KeywordBaseline(), "single", case_ids=ids)
    save(output / "baseline.json", baseline)
    smoke = run(Post("How can I teach my dog to sit using positive reinforcement?"), capped, "two-pass")
    save(output / "smoke.json", smoke)
    reports = {"baseline": baseline}
    status = "complete"
    if smoke["status"] != "ok":
        status = "smoke_failed"
    else:
        for mode in ("single", "two-pass"):
            report = evaluate(data, "dev", capped, mode, stop_on_error=True, case_ids=ids)
            report["latency_includes_rate_pacing"] = True
            reports[mode] = report
            save(output / (mode + ".json"), report)
            if not report["complete"]:
                status = mode + "_failed"
                break
    tokens = Counter()
    for trace in smoke["traces"]:
        tokens.update(trace["usage"])
    for name, report in reports.items():
        if name != "baseline":
            tokens.update(report["tokens"])
    summary = {
        "status": status, "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": os.environ.get("GITHUB_SHA"),
        "provider": provider.name, "model": provider.model,
        "generation_config": {"temperature": 0, "thinking_budget": 0, "max_output_tokens": 1200},
        "free_tier_confirmed_by_operator": True, "billing_tier_verified_by_code": False,
        "api_attempts": capped.calls, "api_attempt_cap": MAX_CALLS,
        "min_request_interval_seconds": getattr(provider, "min_interval_seconds", None),
        "latency_includes_rate_pacing": True,
        "synthetic_data": True, "dataset_sha256": DATA_SHA256,
        "split": "dev", "batch": batch, "total_batches": 4,
        "selected_case_ids": ids, "test_split_used": False,
        "live_api_response_verified": provider.name == "gemini-live" and smoke["status"] == "ok",
        "tokens_observed": dict(tokens),
        "metrics": {name: report["metrics"] for name, report in reports.items()},
        "warning": "Only five synthetic development cases, not a benchmark. Labels are provisional. Billing tier is confirmed by the operator, not verified by this code. Quotas may stop a batch. No retries or paid fallback. Latency includes pacing. Model aliases may change.",
    }
    save(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gemini synthetic evaluation; at most 17 API attempts per batch")
    parser.add_argument("--allow-api-calls", action="store_true")
    parser.add_argument("--confirm-free-tier", action="store_true")
    parser.add_argument("--batch", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if not args.allow_api_calls or not args.confirm_free_tier:
        parser.error("Both --allow-api-calls and --confirm-free-tier are required; verify the project has billing disabled")
    try:
        provider = GeminiProvider()
        summary = execute(Path(__file__).parents[1] / "data/cases.jsonl", args.output_dir,
                          provider, args.batch, confirm_free_tier=True)
    except (ProviderError, ValueError, OSError):
        print("Setup failed. Check the Gemini secret, synthetic fixtures and output directory privately. No secrets logged.")
        return 2
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
