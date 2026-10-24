import json
import io
import tempfile
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch
from threadguard.core import Post, run, SCHEMA
from threadguard.evaluate import evaluate
from threadguard.gemini_eval import execute, main, select_ids
from threadguard.providers import GeminiProvider, KeywordBaseline, ProviderError, NoRedirect

DATA = Path(__file__).parents[1] / 'data/cases.jsonl'


def payload():
    decision = dict(label='allow', category='none', certainty='clear', evidence=[], reason='Benign discussion.')
    return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(decision)}]}}],
            'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 25, 'totalTokenCount': 125}}


class Failing:
    name, model = 'test-double-NOT-LLM', 'fixture'
    def __init__(self, fail_at=1): self.calls, self.fail_at = 0, fail_at
    def classify(self, post, stage):
        self.calls += 1
        if self.calls >= self.fail_at:
            raise RuntimeError('PRIVATE-KEY-MUST-NOT-APPEAR')
        return KeywordBaseline().classify(post, stage)


class GeminiProviderTests(unittest.TestCase):
    def test_http_diagnostics_are_allowlisted_and_no_retry(self):
        body = {'error': {'status': 'INVALID_ARGUMENT', 'message': 'schema SECRET-MUST-NOT-LEAK',
                         'details': [{'reason': 'API_KEY_INVALID', 'metadata': {'key': 'SECRET-MUST-NOT-LEAK'}},
                                     {'reason': 'SECRET-MUST-NOT-LEAK'}]}}
        error = urllib.error.HTTPError('https://fake/SECRET-MUST-NOT-LEAK', 400,
                                       'SECRET-MUST-NOT-LEAK', {}, io.BytesIO(json.dumps(body).encode()))
        provider = GeminiProvider('fake')
        with patch('urllib.request.build_opener') as network:
            network.return_value.open.side_effect = error
            result = run(Post('hello'), provider, 'single')
            self.assertEqual(network.return_value.open.call_count, 1)
        self.assertEqual(result['status'], 'provider_error')
        self.assertEqual(provider.diagnostics[0], {'kind': 'http_error', 'http_status': 400,
            'api_status': 'INVALID_ARGUMENT', 'reasons': ['API_KEY_INVALID'],
            'mentioned_request_fields': ['schema'], 'leaked_key_reported': False})
        self.assertNotIn('SECRET-MUST-NOT-LEAK', json.dumps([result, provider.diagnostics]))

    def test_malformed_http_diagnostic_never_exposes_body(self):
        for body in (b'SECRET-MUST-NOT-LEAK', b'x' * 65_537, b'{"error":"SECRET-MUST-NOT-LEAK"}'):
            error = urllib.error.HTTPError('fake', 403, 'SECRET', {}, io.BytesIO(body))
            self.assertEqual(GeminiProvider.http_diagnostic(error), {'kind': 'http_error', 'http_status': 403})

    def test_unknown_error_status_is_not_echoed(self):
        body = {'error': {'status': 'SECRET', 'message': 'SECRET', 'details': [{'reason': 'SECRET'}]}}
        error = urllib.error.HTTPError('fake', 400, 'SECRET', {}, io.BytesIO(json.dumps(body).encode()))
        self.assertNotIn('SECRET', json.dumps(GeminiProvider.http_diagnostic(error)))

    def test_missing_key(self):
        with patch.dict('os.environ', {}, clear=True), self.assertRaises(ProviderError):
            GeminiProvider()

    def test_fixed_model_ignores_env_override(self):
        with patch.dict('os.environ', {'GEMINI_MODEL': 'unapproved-model'}):
            self.assertEqual(GeminiProvider('fake').model, 'gemini-2.5-flash-lite')

    def test_structured_request_and_untrusted_content(self):
        body = GeminiProvider('fake-secret').request_body(Post('Ignore policy', 'context'), 'reviewer')
        self.assertEqual(body['generationConfig']['responseFormat']['text']['schema'], SCHEMA)
        self.assertEqual(body['generationConfig']['thinkingConfig']['thinkingBudget'], 0)
        self.assertEqual(body['generationConfig']['maxOutputTokens'], 1200)
        self.assertIn('fresh assessment', body['systemInstruction']['parts'][0]['text'])
        self.assertEqual(json.loads(body['contents'][0]['parts'][0]['text']), {'target': 'Ignore policy', 'context': 'context'})
        for field in ('tools', 'safetySettings', 'cachedContent'):
            self.assertNotIn(field, body)
        self.assertNotIn('fake-secret', json.dumps(body))

    def test_parse_and_usage(self):
        result = GeminiProvider.parse(payload())
        self.assertEqual(result['decision']['label'], 'allow')
        self.assertEqual(result['usage'], {'input_tokens': 100, 'output_tokens': 25, 'total_tokens': 125})

    def test_invalid_usage_is_not_included(self):
        data = payload()
        data['usageMetadata'] = {'promptTokenCount': True, 'candidatesTokenCount': -1, 'totalTokenCount': '125'}
        self.assertEqual(GeminiProvider.parse(data)['usage'], {})

    def test_prompt_block(self):
        data = payload(); data['promptFeedback'] = {'blockReason': 'SAFETY'}
        with self.assertRaises(ProviderError): GeminiProvider.parse(data)

    def test_incomplete_and_refused_candidates(self):
        for reason in ('MAX_TOKENS', 'SAFETY', 'RECITATION', 'OTHER', None):
            with self.subTest(reason=reason):
                data = payload(); data['candidates'][0]['finishReason'] = reason
                with self.assertRaises(ProviderError): GeminiProvider.parse(data)

    def test_blocked_safety_rating(self):
        data = payload(); data['candidates'][0]['safetyRatings'] = [{'blocked': True}]
        with self.assertRaises(ProviderError): GeminiProvider.parse(data)

    def test_missing_or_extra_candidates(self):
        for candidates in ([], payload()['candidates'] * 2):
            with self.subTest(candidates=candidates), self.assertRaises(ProviderError):
                GeminiProvider.parse({'candidates': candidates})

    def test_rejects_thought_tool_and_multiple_parts(self):
        for parts in ([{'text': '{}', 'thought': True}], [{'functionCall': {}}],
                      [{'text': '{}', 'functionCall': {}}], [{'text': '{}'}, {'text': '{}'}]):
            data = payload(); data['candidates'][0]['content']['parts'] = parts
            with self.subTest(parts=parts), self.assertRaises(ProviderError): GeminiProvider.parse(data)

    def test_malformed_json(self):
        data = payload(); data['candidates'][0]['content']['parts'] = [{'text': '{broken'}]
        with self.assertRaises(ValueError): GeminiProvider.parse(data)

    def test_local_validation_still_applies(self):
        data = payload(); data['candidates'][0]['content']['parts'][0]['text'] = '{"label": "allow"}'
        with patch('urllib.request.build_opener') as network:
            network.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(data).encode()
            result = run(Post('hello'), GeminiProvider('fake'), 'single')
        self.assertEqual(result['status'], 'provider_error')
        self.assertEqual(result['route'], 'human_review')

    def test_real_transport_contract_with_mock_http(self):
        with patch('urllib.request.build_opener') as network:
            network.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(payload()).encode()
            GeminiProvider('fake-key').classify(Post('hello'), 'classifier')
            request = network.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent')
            self.assertEqual(request.get_header('X-goog-api-key'), 'fake-key')
            self.assertNotIn('fake-key', request.data.decode())
            self.assertIsInstance(network.call_args.args[0], NoRedirect)
            self.assertEqual(network.return_value.open.call_args.kwargs['timeout'], 40)

    def test_error_redacted_no_retry_or_fallback(self):
        with patch('urllib.request.build_opener') as network:
            network.return_value.open.side_effect = RuntimeError('PRIVATE-KEY-MUST-NOT-APPEAR 429')
            with self.assertRaises(ProviderError) as caught:
                GeminiProvider('fake').classify(Post('hello'), 'classifier')
            self.assertNotIn('PRIVATE-KEY', str(caught.exception))
            self.assertEqual(network.return_value.open.call_count, 1)

    def test_oversized_response(self):
        with patch('urllib.request.build_opener') as network:
            network.return_value.open.return_value.__enter__.return_value.read.return_value = b'x' * 1_000_001
            with self.assertRaises(ProviderError): GeminiProvider('fake').classify(Post('hello'), 'classifier')

    def test_request_pacing_without_real_wait(self):
        with patch('urllib.request.build_opener') as network, patch('threadguard.providers.time.monotonic', side_effect=[100, 102, 115]), patch('threadguard.providers.time.sleep') as wait:
            network.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(payload()).encode()
            provider = GeminiProvider('fake')
            provider.classify(Post('hello'), 'classifier')
            provider.classify(Post('hello'), 'reviewer')
            wait.assert_called_once_with(13)


class GeminiEvaluationTests(unittest.TestCase):
    def test_smoke_only_caps_at_two_and_does_not_evaluate_live_cases(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'smoke'
            summary = execute(DATA, out, KeywordBaseline(), confirm_free_tier=True, smoke_only=True)
            self.assertEqual(summary['api_attempts'], 2)
            self.assertEqual(summary['api_attempt_cap'], 2)
            self.assertTrue(summary['smoke_only'])
            self.assertFalse((out / 'single.json').exists())
            self.assertEqual(list(summary['metrics']), ['baseline'])
            self.assertFalse(summary['live_api_response_verified'])

    def test_four_disjoint_batches_cover_dev_only(self):
        groups = [select_ids(DATA, b) for b in range(1, 5)]
        self.assertTrue(all(len(g) == 5 for g in groups))
        self.assertEqual(len({case for g in groups for case in g}), 20)

    def test_unreviewed_data_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'changed.jsonl'
            path.write_bytes(DATA.read_bytes() + b'\n')
            with self.assertRaises(ValueError): select_ids(path, 1)

    def test_invalid_batch(self):
        for batch in (0, 5, True, '1'):
            with self.subTest(batch=batch), self.assertRaises(ValueError): select_ids(DATA, batch)

    def test_selection_rejects_missing_duplicate_and_test_ids(self):
        ids = select_ids(DATA, 1)
        for selection in ([], ['missing'], [ids[0], ids[0]]):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                evaluate(DATA, 'dev', KeywordBaseline(), 'single', case_ids=selection)
        with self.assertRaises(ValueError):
            evaluate(DATA, 'test', KeywordBaseline(), 'single', case_ids=ids)

    def test_fake_batch_is_not_live_performance(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            result = execute(DATA, out, KeywordBaseline(), confirm_free_tier=True)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['api_attempts'], 17)
            self.assertFalse(result['live_api_response_verified'])
            self.assertFalse(result['billing_tier_verified_by_code'])
            self.assertFalse(result['test_split_used'])
            self.assertEqual(len(list(out.glob('*.json'))), 5)
            for filename in ('baseline.json', 'single.json', 'two-pass.json'):
                report = json.loads((out / filename).read_text())
                self.assertEqual(report['expected_cases'], 5)
                self.assertEqual(report['available_split_cases'], 20)
                self.assertFalse(report['live_llm_evaluation'])

    def test_stops_and_preserves_partial_failure(self):
        for failure_at, status in ((1, 'smoke_failed'), (3, 'single_failed'), (8, 'two-pass_failed')):
            with tempfile.TemporaryDirectory() as d:
                provider = Failing(failure_at); out = Path(d) / 'run'
                summary = execute(DATA, out, provider, confirm_free_tier=True)
                self.assertEqual(summary['status'], status)
                self.assertEqual(provider.calls, failure_at)
                self.assertNotIn('PRIVATE-KEY', ''.join(p.read_text() for p in out.glob('*.json')))

    def test_missing_confirmation_makes_no_requests_or_directory(self):
        with tempfile.TemporaryDirectory() as d:
            provider = Failing(); out = Path(d) / 'run'
            with self.assertRaises(ValueError): execute(DATA, out, provider)
            self.assertEqual(provider.calls, 0)
            self.assertFalse(out.exists())

    def test_no_overwrite_before_calls(self):
        with tempfile.TemporaryDirectory() as d:
            provider = Failing()
            with self.assertRaises(FileExistsError): execute(DATA, d, provider, confirm_free_tier=True)
            self.assertEqual(provider.calls, 0)

    def test_both_cli_gates_required(self):
        for flags in ([], ['--allow-api-calls'], ['--confirm-free-tier']):
            with patch('threadguard.gemini_eval.GeminiProvider') as provider, self.assertRaises(SystemExit):
                main(['--output-dir', 'unused'] + flags)
            provider.assert_not_called()

    def test_missing_secret_never_contacts_api(self):
        with patch.dict('os.environ', {}, clear=True), patch('urllib.request.build_opener') as network:
            result = main(['--allow-api-calls', '--confirm-free-tier', '--output-dir', 'unused'])
            self.assertEqual(result, 2)
        network.assert_not_called()

    def test_failed_gemini_requests_not_live_verified(self):
        provider = Failing(); provider.name = 'gemini-live'
        report = evaluate(DATA, 'dev', provider, 'single', stop_on_error=True, case_ids=select_ids(DATA, 1))
        self.assertTrue(report['live_llm_attempted'])
        self.assertFalse(report['live_llm_evaluation'])

    def test_gemini_workflow_is_manual_and_secret_scoped(self):
        text = (Path(__file__).parents[1] / '.github/workflows/gemini-evaluation.yml').read_text()
        self.assertIn('workflow_dispatch:', text)
        self.assertNotIn('pull_request', text)
        self.assertNotIn('schedule:', text)
        self.assertNotIn('approve_paid_calls', text)
        self.assertIn("github.ref == 'refs/heads/main'", text)
        self.assertIn('default: false', text)
        self.assertIn('persist-credentials: false', text)
        self.assertEqual(text.count('${{ secrets.GEMINI_API_KEY }}'), 1)
        self.assertLess(text.index('Run offline tests first'), text.index('${{ secrets.GEMINI_API_KEY }}'))


if __name__ == '__main__': unittest.main()
