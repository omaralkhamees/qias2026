"""
Mistral provider — wraps the Mistral Python SDK.

Requires:
  - MISTRAL_API_KEY in .env or environment
  - pip install mistralai python-dotenv
"""

import os
import time

from dotenv import load_dotenv
from mistralai import Mistral

from src.providers.base import LLMProvider

load_dotenv()


def _is_repetition_loop(text: str, min_chunk: int = 8, min_repeats: int = 20) -> bool:
    """Detect if the model output is stuck in a repetition loop.

    Checks whether any short chunk of text (8-80 chars) is repeated 20+ times
    consecutively, which indicates the model is looping.
    """
    if len(text) < 500:
        return False
    # Check the last 2000 chars for repeated patterns
    tail = text[-2000:]
    for chunk_len in (8, 16, 32, 64):
        pattern = tail[-chunk_len:]
        if not pattern.strip():
            continue
        count = tail.count(pattern)
        if count >= min_repeats:
            return True
    return False


class MistralProvider(LLMProvider):

    def __init__(self, config: dict):
        super().__init__(config)
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env or environment")
        self.client = Mistral(api_key=api_key)

    def call(self, system_prompt: str, question: str, label: str = "") -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                )
                text = response.choices[0].message.content
                if text is None:
                    raise ValueError("Mistral returned empty response (no content)")

                if _is_repetition_loop(text):
                    raise ValueError("Model stuck in repetition loop")

                return text

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"    [{label}] Attempt {attempt+1} failed ({e}). Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"All {self.max_retries} attempts failed: {e}") from e

        raise RuntimeError("Unreachable")
