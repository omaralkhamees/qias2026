"""
Abstract base class for LLM providers.

To add a new provider:
  1. Create a new file in src/providers/ (e.g. openai.py)
  2. Subclass LLMProvider and implement `call()`
  3. Register it in PROVIDER_REGISTRY at the bottom of this file
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self, config: dict):
        """
        Args:
            config: The `model` section from config.yaml
        """
        self.model_name = config["name"]
        self.temperature = config.get("temperature", 0)
        self.max_output_tokens = config.get("max_output_tokens", 16384)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5)

    @abstractmethod
    def call(self, system_prompt: str, question: str, label: str = "") -> str:
        """
        Send a question to the LLM and return the raw text response.

        Args:
            system_prompt: The full system prompt text
            question: The inheritance question in Arabic
            label: Optional label for logging (e.g. "03/200")

        Returns:
            Raw text response from the model
        """
        ...


# ── Provider Registry ─────────────────────────────────────────
# Import providers here to avoid circular imports.
# Each provider file registers itself by being imported.

def get_provider(config: dict) -> LLMProvider:
    """Factory: instantiate the correct provider from config."""
    name = config["provider"]

    if name == "gemini":
        from src.providers.gemini import GeminiProvider
        return GeminiProvider(config)
    elif name == "mistral":
        from src.providers.mistral_provider import MistralProvider
        return MistralProvider(config)
    else:
        raise ValueError(
            f"Unknown provider '{name}'. "
            f"Available: gemini, mistral. "
            f"Add new providers in src/providers/"
        )
