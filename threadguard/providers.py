"""Real Responses API adapter and an explicitly non-LLM demonstration baseline."""
import json
import os
import re
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

    def __init__(self, model=None, key=None):
        self.model = model or os.environ.get("OPENAI_MODEL", "")
        self._key = key or os.environ.get("OPENAI_API_KEY", "")
        if not self._key or not self.model:
            raise ProviderError("Configure OPENAI_API_KEY and OPENAI_MODEL securely; do not paste keys into chat.")

    def request_body(self, post, stage):
        instruction = POLICY
        if stage == "reviewer":
            instruction += "\nMake a fresh assessment. Pay particular attention to quotation, missing context, promotional relevance, and false positives."
        return {
            "model": self.model, "store": False, "max_output_tokens": 1200,
            "input": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({"target": post.target, "context": post.context}, ensure_ascii=False)},
            ],
            "text": {"format": {"type": "json_schema", "name": "moderation_decision", "strict": True, "schema": SCHEMA}},
        }

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
