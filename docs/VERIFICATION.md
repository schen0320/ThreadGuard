# Verification record

Initial local verification: October 24, 2026. Environment: Python 3.12, local Codex workspace. This record separates local test evidence from later GitHub checks.

## Completed

* `python -m unittest discover -s tests -v`: 39 tests passed. These use synthetic inputs and fake provider responses. They test schema validation, exact quote checking, bounded routing, disagreement, errors, refusals, prompt construction, dataset separation and metrics.
* `python -m threadguard evaluate --provider baseline --mode single --split dev --output reports/baseline-dev.json`: completed on 20 development fixtures. Three-label accuracy 65%, flag precision on known binary labels 77.78%, flag recall 87.5%, false-positive rate 25%. These are **keyword-baseline results, not LLM performance**. See the report for denominators, confusion matrix, provenance and hashes.
* 40 authored synthetic cases exist, split 20 development / 20 test. All labels are provisional; no independent human annotation has been completed. The test split has not been evaluated against a live model.

## Not completed

* No real LLM requests, token bill, latency benchmark or LLM accuracy measurement.
* No model comparison, prompt improvement validated on unseen data, or production traffic.
* No public application deployment. Repository publication does not deploy a service or run live model inference.

An API credential and a supported model must be configured securely before a live run. Check [GitHub Actions](https://github.com/schen0320/ThreadGuard/actions) for remote test results associated with each published commit. A passing offline CI run is not evidence of live model performance.

## Manual live-evaluation setup

October 24, 2026: added a manually triggered, secret-backed evaluation workflow and bounded runner. The expanded offline suite passes 50 tests locally, including request limits, stopping on API errors, secret-safe error reporting, credential preflight, and preserving previous reports. Its 62-call maximum includes the smoke test and both development comparisons; the test split is not used. No real model responses have been verified as part of this setup. A GitHub secret and a successful manual run are still needed.

## Gemini free-tier setup

October 24, 2026: added the Gemini adapter and a separate manual workflow without changing the paid OpenAI runner. **78 offline tests pass locally.** The 28 additional tests use mocked HTTP and fake providers, not real Gemini calls. They cover request/schema construction, key placement in headers, secret-redacted errors, safety refusals, incomplete or malformed output, local evidence validation, request pacing, batch selection, data-file integrity, failure preservation, billing-tier confirmation, and missing-key behavior.

The runner uses a fixed `gemini-2.5-flash-lite` model, five development cases per batch, and at most 17 API attempts including smoke testing and both evaluation modes. Four disjoint batches cover the 20 development cases; no live evaluation of the test split has occurred. The code has no paid fallback, retry, or billing configuration mechanism. The operator must confirm the key belongs to a Free Tier project with billing disabled; code does not verify account billing status. Rate-limit pacing is included in reported latency.

At this verification point, `GEMINI_API_KEY` is not configured in the local runtime, and no Gemini response or accuracy result has been verified. The adapter's HTTP contract and safety behavior are only covered by offline tests until the first real run succeeds. Do not present the 78 passing tests as LLM performance.

## First Gemini attempt and diagnostic hardening

October 24, 2026: the secret was configured in GitHub Actions, not the local runtime. [Run 40084117530](https://github.com/schen0320/ThreadGuard/actions/runs/40084117530), at source commit `ef1bdafe2435b299425b5f043412f973817724e2`, passed all 78 offline tests but failed on its first smoke request. It stopped after one API attempt with no verified model output or token usage. The baseline, failed smoke and summary artifacts are preserved by that run. Its overly broad error redaction does not establish whether the cause was access, request format, quota, network or response parsing.

Added opt-in smoke-only execution (maximum two requests) and bounded HTTP diagnostics that return only integer status codes and fixed allowlisted labels. Raw errors, headers, credentials and server messages are never exported. No retry, paid fallback or permission/billing changes are introduced. These changes enable diagnosis; they do not themselves establish working inference.
