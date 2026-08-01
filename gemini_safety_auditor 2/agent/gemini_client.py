"""
Thin wrapper around the Gemini API.

Deliberately minimal -- two functions, generate() and embed() -- so the
rest of the pipeline (probes, attack loop, orchestrator) doesn't need to
know anything about HTTP or the Gemini request format. If you swap in the
official `google-generativeai` SDK later, only this file changes.
"""

import time
import requests

from .config import require_api_key, API_BASE, GENERATION_MODEL, EMBEDDING_MODEL


class GeminiError(RuntimeError):
    pass


def generate(prompt: str, image_bytes: bytes | None = None,
             image_mime: str = "image/jpeg", retries: int = 3,
             model: str | None = None, api_key: str | None = None) -> dict:
    """
    Calls the Gemini generateContent endpoint with a text prompt and an
    optional image. Returns a dict with:
        text            -- the model's text output (may be empty if refused)
        refused         -- bool, best-effort guess at whether this was a refusal
        finish_reason   -- raw finishReason field from the API
        raw             -- full raw response JSON, for logging/debugging

    `model` overrides config.GENERATION_MODEL for this call only -- used by
    agent/cross_tier.py to run the same prompt against two different tiers
    without needing a second copy of this function.

    `api_key` overrides the env-var-derived key for this call only -- used
    by the Streamlit app so each visitor's own key is used explicitly
    rather than relying on a shared process-wide env var.
    """
    api_key = require_api_key(api_key)
    use_model = model or GENERATION_MODEL
    url = f"{API_BASE}/models/{use_model}:generateContent?key={api_key}"

    parts = [{"text": prompt}]
    if image_bytes is not None:
        import base64
        parts.append({
            "inline_data": {
                "mime_type": image_mime,
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            }
        })

    payload = {"contents": [{"parts": parts}]}

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return _parse_generate_response(data)
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))  # backoff for rate limits

    raise GeminiError(f"generate() failed after {retries} attempts: {last_err}")


def _parse_generate_response(data: dict) -> dict:
    candidates = data.get("candidates", [])
    if not candidates:
        # Blocked entirely at the prompt level (safety filter fired before
        # generation even started) -- treat this as the strongest form of refusal.
        return {
            "text": "",
            "refused": True,
            "finish_reason": data.get("promptFeedback", {}).get("blockReason", "PROMPT_BLOCKED"),
            "raw": data,
        }

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason", "")
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)

    refused = finish_reason in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT")
    return {"text": text, "refused": refused, "finish_reason": finish_reason, "raw": data}


def embed(text: str, retries: int = 3, api_key: str | None = None) -> list[float]:
    """
    Returns the embedding vector for a piece of text using Gemini's
    embedding model. This is our stand-in for "hidden layer activations"
    since a hosted API doesn't expose real internals -- see probes/embedding_probe.py
    for why this is an honest but weaker proxy than true activation access.

    `api_key` overrides the env-var-derived key for this call only.
    """
    api_key = require_api_key(api_key)
    url = f"{API_BASE}/models/{EMBEDDING_MODEL}:embedContent?key={api_key}"
    payload = {"content": {"parts": [{"text": text}]}}

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["embedding"]["values"]
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))

    raise GeminiError(f"embed() failed after {retries} attempts: {last_err}")
