"""
Gemini provider — wraps Google Generative AI SDK.

Requires:
  - GEMINI_API_KEY in .env or environment
  - pip install google-genai python-dotenv
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.providers.base import LLMProvider

load_dotenv()


class GeminiProvider(LLMProvider):

    def __init__(self, config: dict):
        super().__init__(config)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env or environment")
        self.client = genai.Client(api_key=api_key)
        self.tpm_limit = config.get("tpm_limit", 0)
        self._last_call_time = 0.0
        self._last_call_tokens = 0

    def _wait_for_rate_limit(self):
        """Sleep if needed to stay within the TPM budget."""
        if self.tpm_limit <= 0 or self._last_call_tokens == 0:
            return
        # Time we need to "reserve" for the last call's tokens
        required_wait = (self._last_call_tokens / self.tpm_limit) * 60.0
        elapsed = time.time() - self._last_call_time
        remaining = required_wait - elapsed
        if remaining > 0:
            print(f"    [rate-limit] {self._last_call_tokens} tokens used, waiting {remaining:.0f}s ...", flush=True)
            time.sleep(remaining)

    def call(self, system_prompt: str, question: str, label: str = "") -> str:
        self._wait_for_rate_limit()

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                    contents=question,
                )

                # Track token usage for rate limiting
                if self.tpm_limit > 0 and response.usage_metadata:
                    self._last_call_tokens = response.usage_metadata.total_token_count or 0
                    self._last_call_time = time.time()

                # Extract text — handle different response shapes
                text = None
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    text = response.candidates[0].content.parts[-1].text
                if text is None:
                    text = response.text
                if text is None:
                    raise ValueError("Gemini returned empty response (no text)")
                return text

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"    [{label}] Attempt {attempt+1} failed ({e}). Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"All {self.max_retries} attempts failed: {e}") from e

        raise RuntimeError("Unreachable")