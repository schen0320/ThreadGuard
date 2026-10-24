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
