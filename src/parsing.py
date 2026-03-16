"""
Parsing utilities for model responses.

Handles:
  - Extracting Arabic reasoning from <تفكير> tags
  - Extracting structured JSON from markdown code blocks or raw JSON
"""

import json
import re


def extract_reasoning(text: str) -> str:
    """Extract reasoning text inside <تفكير> tags, or return full text if no tags."""
    if not text:
        return ""
    match = re.search(r"<تفكير>(.*?)</تفكير>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_json(text: str) -> dict | None:
    """
    Extract structured answer from model response.

    Tries in order:
      1. JSON inside ```json ... ``` code blocks (after stripping <تفكير> tags)
      2. First { ... } block in the remaining text

    If the parsed JSON contains an "answer_structured" key, unwraps it.
    Returns None if no valid JSON found.
    """
    if not text:
        return None

    # Remove reasoning tags so we only parse the answer portion
    stripped = re.sub(r"<تفكير>.*?</تفكير>", "", text, flags=re.DOTALL)

    # Normalize non-breaking spaces (U+00A0) to regular spaces —
    # some models (e.g. Mistral) emit \xa0 in JSON indentation,
    # which is not valid JSON whitespace and breaks json.loads().
    stripped = stripped.replace("\xa0", " ")

    # Try ```json ... ``` first
    match = re.search(r"```json\s*(.*?)\s*```", stripped, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return parsed.get("answer_structured", parsed)
        except json.JSONDecodeError:
            pass

    # Fallback: first JSON object in text
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed.get("answer_structured", parsed)
        except json.JSONDecodeError:
            pass

    return None
