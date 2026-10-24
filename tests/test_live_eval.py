import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from threadguard.core import Post
from threadguard.evaluate import evaluate
from threadguard.live_eval import CappedProvider, execute, main
from threadguard.providers import KeywordBaseline, OpenAIProvider, ProviderError

DATA = Path(__file__).parents[1] / 'data/cases.jsonl'


class Failing:
    name, model = 'test-double-NOT-LLM', 'fixture'
    def __init__(self): self.calls = 0
    def classify(self, post, stage):
        self.calls += 1
        raise RuntimeError('DO-NOT-LOG-THIS-SECRET')


class LiveEvaluationTests(unittest.TestCase):
    def test_request_configuration(self):
        p = OpenAIProvider('gpt-6-luna', 'fake', 'none')
        body = p.request_body(Post('hello'), 'classifier')
        self.assertEqual(body['reasoning']['effort'], 'none')
        self.assertEqual(body['service_tier'], 'default')
        self.assertEqual(body['max_output_tokens'], 1200)

    def test_invalid_effort_rejected(self):
        with self.assertRaises(ProviderError): OpenAIProvider('fixture', 'fake', 'invalid')

    def test_call_cap(self):
        p = CappedProvider(KeywordBaseline(), max_calls=1)
        p.classify(Post('hello'), 'classifier')
        with self.assertRaises(ProviderError): p.classify(Post('hello'), 'reviewer')
        self.assertEqual(p.calls, 1)

    def test_errors_count_toward_cap(self):
        p = CappedProvider(Failing(), max_calls=1)
        with self.assertRaises(RuntimeError): p.classify(Post('hello'), 'classifier')
        with self.assertRaises(ProviderError): p.classify(Post('hello'), 'classifier')
        self.assertEqual(p.inner.calls, 1)

    def test_evaluation_stops_after_first_error(self):
        p = Failing()
        report = evaluate(DATA, 'dev', p, 'single', stop_on_error=True)
        self.assertEqual(p.calls, 1)
        self.assertEqual(report['expected_cases'], 20)
        self.assertEqual(report['completed_cases'], 1)
        self.assertFalse(report['complete'])
        self.assertEqual(report['metrics']['provider_errors'], 1)

    def test_failed_requests_not_verified_as_live(self):
        p = Failing()
        p.name = 'openai-live'
        report = evaluate(DATA, 'dev', p, 'single', stop_on_error=True)
        self.assertTrue(report['live_llm_attempted'])
        self.assertFalse(report['live_llm_evaluation'])

    def test_full_runner_with_fake_provider(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            summary = execute(DATA, out, KeywordBaseline())
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['api_attempts'], 62)
            self.assertFalse(summary['test_split_used'])
            self.assertFalse(summary['live_api_response_verified'])
            self.assertEqual(len(list(out.glob('*.json'))), 5)

    def test_failed_smoke_stops_before_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Failing()
            out = Path(d) / 'run'
            summary = execute(DATA, out, p)
            self.assertEqual(p.calls, 1)
            self.assertEqual(summary['status'], 'smoke_failed')
            self.assertEqual(set(summary['metrics']), {'baseline'})
            self.assertNotIn('DO-NOT-LOG', ''.join(f.read_text() for f in out.glob('*.json')))

    def test_never_overwrites_previous_results(self):
        with tempfile.TemporaryDirectory() as d:
            p = Failing()
            with self.assertRaises(FileExistsError): execute(DATA, d, p)
            self.assertEqual(p.calls, 0)

    def test_explicit_consent_required(self):
        with self.assertRaises(SystemExit), patch('threadguard.live_eval.OpenAIProvider') as provider:
            main(['--output-dir', 'unused'])
        provider.assert_not_called()

    def test_missing_key_exits_without_network(self):
        with patch.dict('os.environ', {}, clear=True), patch('urllib.request.build_opener') as network:
            self.assertEqual(main(['--allow-api-calls', '--output-dir', 'unused']), 2)
        network.assert_not_called()


if __name__ == '__main__': unittest.main()
