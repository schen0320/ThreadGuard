"""Manual, bounded evaluation on synthetic development data only."""
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .core import Post, run
from .evaluate import evaluate, load_cases
from .providers import KeywordBaseline, OpenAIProvider, ProviderError

MODEL = "gpt-6-luna"
MAX_CALLS = 62  # Two smoke requests + 20 single-pass + 40 two-pass.


class CappedProvider:
    def __init__(self, inner, max_calls=MAX_CALLS):
        self.inner = inner
        self.name, self.model = inner.name, inner.model
        self.max_calls, self.calls = max_calls, 0

    def classify(self, post, stage):
        if self.calls >= self.max_calls:
            raise ProviderError("Request cap reached")
        self.calls += 1  # Count attempts, including errors; never retry.
        return self.inner.classify(post, stage)


def save(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def execute(data, output, provider):
    cases = load_cases(data, "dev")
    if len(cases) != 20:
        raise ValueError("This bounded run requires exactly 20 development fixtures")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    capped = CappedProvider(provider)
    started = datetime.now(timezone.utc).isoformat()
    baseline = evaluate(data, "dev", KeywordBaseline(), "single")
    save(output / "baseline.json", baseline)
    smoke = run(Post("How can I teach my dog to sit using positive reinforcement?"), capped, "two-pass")
    save(output / "smoke.json", smoke)
    reports = {"baseline": baseline}
    status = "complete"
    if smoke["status"] != "ok":
        status = "smoke_failed"
    else:
        for mode in ("single", "two-pass"):
            report = evaluate(data, "dev", capped, mode, stop_on_error=True)
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
        "reasoning_effort": getattr(provider, "reasoning_effort", None),
        "max_output_tokens_per_request": 1200, "service_tier": "default",
        "api_attempts": capped.calls, "api_attempt_cap": MAX_CALLS,
        "synthetic_data": True, "split": "dev", "test_split_used": False,
        "live_api_response_verified": provider.name == "openai-live" and smoke["status"] == "ok",
        "tokens_observed": dict(tokens),
        "metrics": {name: report["metrics"] for name, report in reports.items()},
        "warning": "Provisional synthetic labels, not a blind benchmark. Token totals can omit failed or invalid responses. Not a billing ledger or hard dollar cap. Model aliases may change.",
    }
    save(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manual live development evaluation; at most 62 paid API attempts")
    parser.add_argument("--allow-api-calls", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if not args.allow_api_calls:
        parser.error("Explicit --allow-api-calls consent is required")
    # A fixed model/config prevents an unreviewed workflow input from changing cost.
    try:
        provider = OpenAIProvider(model=MODEL, reasoning_effort="none")
        summary = execute(Path(__file__).parents[1] / "data/cases.jsonl", args.output_dir, provider)
    except (ProviderError, ValueError, OSError):
        print("Setup failed. Check the API secret and output directory privately. No secret values are logged.")
        return 2
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
