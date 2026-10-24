"""Real API adapters and an explicitly non-LLM demonstration baseline."""
import json
import os
import re
import time
import urllib.error
import urllib.request
from .core import POLICY, SCHEMA


class ProviderError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("redirect rejected")


class OpenAIProvider:
    name = "openai-live"

    def __init__(self, model=None, key=None, reasoning_effort=None):
        self.model = model or os.environ.get("OPENAI_MODEL", "")
        self._key = key or os.environ.get("OPENAI_API_KEY", "")
        if not self._key or not self.model:
            raise ProviderError("Configure OPENAI_API_KEY and OPENAI_MODEL securely; do not paste keys into chat.")
        self.reasoning_effort = reasoning_effort or os.environ.get("OPENAI_REASONING_EFFORT", "")
        if self.reasoning_effort not in ("", "none", "low", "medium", "high", "xhigh", "max"):
            raise ProviderError("Invalid reasoning effort")

    def request_body(self, post, stage):
        instruction = POLICY
        if stage == "reviewer":
            instruction += "\nMake a fresh assessment. Pay particular attention to quotation, missing context, promotional relevance, and false positives."
        body = {
            "model": self.model, "store": False, "max_output_tokens": 1200,
            "service_tier": "default",
            "input": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({"target": post.target, "context": post.context}, ensure_ascii=False)},
            ],
            "text": {"format": {"type": "json_schema", "name": "moderation_decision", "strict": True, "schema": SCHEMA}},
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    @staticmethod
    def parse(data):
        if not isinstance(data, dict) or data.get("status") != "completed":
            raise ProviderError("incomplete response")
        texts = []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "refusal":
                    raise ProviderError("model refusal")
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
        if len(texts) != 1:
            raise ProviderError("expected one output")
        raw_usage = data.get("usage", {})
        usage = {k: v for k, v in raw_usage.items() if k in ("input_tokens", "output_tokens", "total_tokens") and type(v) is int and v >= 0}
        return {"decision": json.loads(texts[0]), "usage": usage}

    def classify(self, post, stage):
        body = self.request_body(post, stage)
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key}, method="POST",
        )
        try:
            # No retries: uncertain requests may have already incurred API charges.
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=40) as response:
                raw = response.read(1_000_001)
                if len(raw) > 1_000_000:
                    raise ProviderError("oversized response")
                return self.parse(json.loads(raw))
        except Exception:
            raise ProviderError("API request failed; check access, model, network and billing privately") from None


class GeminiProvider:
    """Fixed free-tier-eligible model; the caller must verify the project's tier.

    There is no API request flag that makes a paid project's calls free.
    Use the bounded gemini_eval runner, not this adapter directly, for the demo.
    """
    name = "gemini-live"
    model = "gemini-2.5-flash-lite"
    min_interval_seconds = 15

    def __init__(self, key=None):
        self._key = key or os.environ.get("GEMINI_API_KEY", "")
        if not self._key:
            raise ProviderError("Configure GEMINI_API_KEY securely; do not paste keys into chat.")
        self._last_started = None
        self.diagnostics = []

    @staticmethod
    def http_diagnostic(error):
        """Only fixed labels and numeric status escape; never error text or headers."""
        result = {"kind": "http_error"}
        if type(error.code) is int and 100 <= error.code <= 599:
            result["http_status"] = error.code
        try:
            raw = error.read(65_537)
            if len(raw) > 65_536:
                return result
            detail = json.loads(raw).get("error", {})
            allowed_status = ("INVALID_ARGUMENT", "UNAUTHENTICATED", "PERMISSION_DENIED",
                              "NOT_FOUND", "RESOURCE_EXHAUSTED", "INTERNAL", "UNAVAILABLE")
            if detail.get("status") in allowed_status:
                result["api_status"] = detail["status"]
            allowed_reasons = ("API_KEY_INVALID", "API_KEY_EXPIRED", "API_KEY_SERVICE_BLOCKED",
                               "API_KEY_HTTP_REFERRER_BLOCKED", "API_KEY_IP_ADDRESS_BLOCKED",
                               "SERVICE_DISABLED", "BILLING_DISABLED", "RATE_LIMIT_EXCEEDED",
                               "ACCESS_TOKEN_SCOPE_INSUFFICIENT", "IAM_PERMISSION_DENIED")
            result["reasons"] = sorted({d.get("reason") for d in detail.get("details", [])
                                         if isinstance(d, dict) and d.get("reason") in allowed_reasons})
            message = detail.get("message", "")
            if isinstance(message, str):
                lowered = message.lower()
                result["mentioned_request_fields"] = [field for field in
                    ("responseFormat", "mimeType", "schema", "thinkingBudget", "maxOutputTokens")
                    if field.lower() in lowered]
                result["leaked_key_reported"] = "leaked" in lowered
        except Exception:
            pass
        return result

    def request_body(self, post, stage):
        instruction = POLICY
        if stage == "reviewer":
            instruction += "\nMake a fresh assessment. Pay particular attention to quotation, missing context, promotional relevance, and false positives."
        return {
            "systemInstruction": {"parts": [{"text": instruction}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(
                {"target": post.target, "context": post.context}, ensure_ascii=False)}]}],
            "generationConfig": {
                "candidateCount": 1, "maxOutputTokens": 1200, "temperature": 0,
                "thinkingConfig": {"thinkingBudget": 0},
                "responseFormat": {"text": {"mimeType": "APPLICATION_JSON", "schema": SCHEMA}},
            },
        }

    @staticmethod
    def parse(data):
        if not isinstance(data, dict) or data.get("error"):
            raise ProviderError("invalid response")
        if data.get("promptFeedback", {}).get("blockReason"):
            raise ProviderError("prompt blocked")
        candidates = data.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ProviderError("expected one candidate")
        candidate = candidates[0]
        if candidate.get("finishReason") != "STOP":
            raise ProviderError("blocked or incomplete response")
        if any(r.get("blocked") for r in candidate.get("safetyRatings", [])):
            raise ProviderError("candidate blocked")
        parts = candidate.get("content", {}).get("parts", [])
        if not isinstance(parts, list) or len(parts) != 1:
            raise ProviderError("expected one text part")
        part = parts[0]
        if (not isinstance(part, dict) or not isinstance(part.get("text"), str)
                or not part["text"] or part.get("thought")
                or set(part) - {"text", "thought", "thoughtSignature"}):
            raise ProviderError("unexpected output type")
        raw_usage = data.get("usageMetadata", {})
        fields = {"promptTokenCount": "input_tokens", "candidatesTokenCount": "output_tokens",
                  "totalTokenCount": "total_tokens"}
        usage = {dest: raw_usage[src] for src, dest in fields.items()
                 if type(raw_usage.get(src)) is int and raw_usage[src] >= 0}
        return {"decision": json.loads(part["text"]), "usage": usage}

    def classify(self, post, stage):
        # No credentials in the URL/body, no tools, no safety overrides or fallback.
        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/" + self.model + ":generateContent",
            data=json.dumps(self.request_body(post, stage)).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self._key}, method="POST",
        )
        try:
            if self._last_started is not None:
                remaining = self.min_interval_seconds - (time.monotonic() - self._last_started)
                if remaining > 0:
                    time.sleep(remaining)
            self._last_started = time.monotonic()
            # No retry, including HTTP 429. Respect project quota; never enable billing.
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=40) as response:
                raw = response.read(1_000_001)
                if len(raw) > 1_000_000:
                    raise ProviderError("oversized response")
                return self.parse(json.loads(raw))
        except urllib.error.HTTPError as error:
            self.diagnostics.append(self.http_diagnostic(error))
            raise ProviderError("Gemini HTTP request failed; see allowlisted diagnostic codes") from None
        except Exception:
            self.diagnostics.append({"kind": "transport_or_output_error"})
            raise ProviderError("Gemini request failed; check key access, model, quota and network privately") from None


class KeywordBaseline:
    """Intentionally weak baseline; never presented as LLM inference."""
    name = "keyword-baseline-NOT-LLM"
    model = "keyword-v1"
    rules = [("threat", r"I will hurt you"), ("harassment", r"you are (?:an idiot|worthless)"),
             ("spam", r"guaranteed profit|claim your prize|buy now")]

    def classify(self, post, stage):
        for category, pattern in self.rules:
            m = re.search(pattern, post.target, re.I)
            if m:
                return {"decision": {"label": "flag", "category": category, "certainty": "clear",
                    "evidence": [m.group()], "reason": "Keyword baseline match; context is not understood."}, "usage": {}}
        return {"decision": {"label": "allow", "category": "none", "certainty": "clear",
            "evidence": [], "reason": "No baseline keywords matched. This is not a safety guarantee."}, "usage": {}}
