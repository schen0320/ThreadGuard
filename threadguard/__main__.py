import argparse
import json
import sys
from pathlib import Path
from .core import Post, run
from .providers import KeywordBaseline, OpenAIProvider, ProviderError
from .evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="ThreadGuard moderation prototype; no automatic enforcement")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("classify", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--provider", choices=["baseline", "openai"], default="baseline")
        p.add_argument("--model")
        p.add_argument("--mode", choices=["single", "two-pass"], default="two-pass")
        p.add_argument("--allow-api-calls", action="store_true", help="Explicit consent to send input to OpenAI and incur API charges")
        if name == "classify":
            p.add_argument("--text", required=True)
            p.add_argument("--context", default="")
        else:
            p.add_argument("--data", default=str(Path(__file__).parents[1] / "data/cases.jsonl"))
            p.add_argument("--split", choices=["dev", "test"], default="dev")
            p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.provider == "openai" and not args.allow_api_calls:
        parser.error("Live inference requires --allow-api-calls; API calls may incur charges.")
    try:
        provider = OpenAIProvider(args.model) if args.provider == "openai" else KeywordBaseline()
        if args.command == "classify":
            result = run(Post(args.text, args.context), provider, args.mode)
        else:
            result = evaluate(args.data, args.split, provider, args.mode)
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Refuse to overwrite an experiment, preserving reproducibility.
            with path.open("x", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            result = {k: v for k, v in result.items() if k not in ("cases", "subgroups")}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ProviderError, ValueError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
