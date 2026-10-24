# ThreadGuard

A small LLM moderation workflow for discussion posts. It assesses spam, harassment and threats, checks exact evidence quotes, and routes uncertain or flagged posts to a human. It is an independent portfolio project, not a Quora product or implementation of Quora's policies.

## Status

Version 0.1 implements a Responses API adapter, bounded two-pass assessment, strict local validation, a keyword baseline and an evaluation harness. The project was developed with Codex assistance. **Live LLM inference has not been verified in the creation environment.** No API credential was configured, no paid model calls were made, and there is no claimed LLM accuracy or production deployment. Offline engineering tests and keyword-baseline results are distinct from model performance.

## Run without credentials

Python 3.10 or later; standard library only. No package installation is needed.

```sh
python -m unittest discover -s tests -v
python -m threadguard classify --text "You are an idiot" --provider baseline
python -m threadguard evaluate --provider baseline --mode single --split dev --output reports/local-dev-baseline.json
```

The baseline is deliberately weak. It misses paraphrases and Chinese text and overflags quotations. It exists to measure whether the LLM approach actually improves anything, not to simulate one.

## Run the real LLM workflow

Use a supported model with Responses API structured-output support. Set `OPENAI_API_KEY` and `OPENAI_MODEL` securely in your local environment or secret manager. Never commit the key, paste it into chat, put it in a URL, or include it in an application. Do not use real customer messages in this demo. The model name is intentionally explicit, not a hardcoded recommendation.

```sh
python -m threadguard classify --provider openai --allow-api-calls --text "A stranger wrote 'I will hurt you'. How do I report it?"
python -m threadguard evaluate --provider openai --allow-api-calls --mode single --split dev --output reports/local-dev-single.json
python -m threadguard evaluate --provider openai --allow-api-calls --mode two-pass --split dev --output reports/local-dev-two-pass.json
```

One post uses at most two model calls. Each request has a 1200 output-token limit and 40-second timeout; reasoning models may need a larger limit after testing. A 20-case two-pass run can make up to 40 billable calls. Configure an account spending limit before running. The client never retries uncertain requests automatically. `store=false` is requested; this is not a claim of zero provider retention. Respect your environment's network policy.

## Workflow

1. Validate target/context size and keep all post content in the untrusted user message.
2. Ask a classifier for a structured label, category, certainty, evidence and short justification.
3. In two-pass mode, ask a reviewer to independently assess the same input without seeing the first answer. The same model is used, so errors can remain correlated.
4. Validate types, category/label consistency and exact evidence substrings locally.
5. Disagreement, refusal, incomplete output or provider error routes to review. An agreed flag is still only a recommendation for a human. There are no deletion, ban or account-action tools.

This is a bounded workflow, not an autonomous enforcement agent. Prompt separation and output validation reduce certain risks but do not prove prompt-injection immunity or semantic correctness. An exact quote can still be irrelevant evidence.

## Evaluation

`data/cases.jsonl` contains 40 AI-assisted synthetic scenarios, 20 development and 20 test cases. They cover benign disagreement, spam, personal insults, threats, quotations, fictional context, prompt injection, ambiguity and Chinese text. These are small illustrative fixtures with provisional labels, not independent ground truth or a blind benchmark. A human should review labels before any performance claim. No Quora/customer data was scraped or copied.

Tune only on development cases. Freeze the policy, prompt and model before using the test split; do not tune on failed test examples and still call them held out. For stronger evidence, have another person label a new unseen set and record disagreements. Test at least two model configurations or seeds/runs and inspect errors manually.

```sh
python -m threadguard evaluate --provider openai --allow-api-calls --mode two-pass --split test --output reports/local-test-frozen.json
```

Reports record dataset and policy hashes, model, timestamp, token usage, per-case outcomes, subgroup metrics, confusion matrix and latency. `flag_recall` counts a review as a missed definite flag. A separate capture metric credits flag-or-review; it must not be substituted for recall. Coverage and selective accuracy are reported together so abstention cannot conceal poor performance. Costs are not guessed: use token totals and verified current provider pricing. Requests that fail before a parsed result can incur charges not represented in token totals; this is not billing reconciliation.

## What to explain in an interview

* Why two passes might reduce some false positives but also raise cost and latency without eliminating correlated errors.
* Why quoted abuse and a direct insult can contain identical words but require different decisions.
* How failure-to-review differs from silently accepting a failed request.
* Why mock tests are engineering evidence, not LLM accuracy, and why this synthetic set needs independent labels.
* What you personally reviewed, changed and tested with AI assistance.

Before adding this as a completed LLM project to a resume, run real inference, inspect every failure, make an improvement you can explain, and record a reproducible evaluation. Do not claim you built or operated Quora's system.

## Repository and automated checks

Source: [schen0320/ThreadGuard](https://github.com/schen0320/ThreadGuard). The included [GitHub Actions workflow](https://github.com/schen0320/ThreadGuard/actions) runs offline tests on pushes and pull requests and needs no secrets. A passing workflow verifies engineering behavior, not live LLM quality. Keep API credentials and real user data out of the repository; local evaluation reports are ignored by Git.

## References

Implementation uses the [OpenAI Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs) checked on October 24, 2026. Refusals and incomplete responses still require explicit handling. Project motivation: [Quora ML Engineer New Grad](https://jobs.ashbyhq.com/quora/40659fe7-545e-4b08-b6d7-2567b18b9ed3).
