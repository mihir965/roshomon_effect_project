"""Live-fetch available chat-capable models from each provider.

Each `list_*` returns a list of model IDs (strings). On any error
(missing key, network) it returns []. Streamlit/CLI use these to populate
pickers and to validate selections before running an evaluation.
"""

from __future__ import annotations

import requests

from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    OLLAMA_BASE_URL,
)


PROVIDERS = ("openai", "anthropic", "gemini", "ollama")


def list_openai_models() -> list[str]:
    if not OPENAI_API_KEY:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        ids = [m.id for m in client.models.list().data]
        chat_prefixes = ("gpt-", "o1", "o3", "o4", "chatgpt-")
        return sorted(m for m in ids if m.startswith(chat_prefixes))
    except Exception:
        return []


def list_anthropic_models() -> list[str]:
    if not ANTHROPIC_API_KEY:
        return []
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return sorted(m.id for m in client.models.list().data)
    except Exception:
        return []


def list_gemini_models() -> list[str]:
    if not GOOGLE_API_KEY:
        return []
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        out: list[str] = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" not in actions:
                continue
            name = m.name or ""
            short = name.split("/", 1)[1] if name.startswith("models/") else name
            out.append(short)
        return sorted(out)
    except Exception:
        return []


def list_ollama_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


_LISTERS = {
    "openai": list_openai_models,
    "anthropic": list_anthropic_models,
    "gemini": list_gemini_models,
    "ollama": list_ollama_models,
}


def list_models(provider: str) -> list[str]:
    return _LISTERS[provider]()


def all_available_models() -> dict[str, list[str]]:
    """Return {provider: [model_id, ...]} for every provider."""
    return {p: list_models(p) for p in PROVIDERS}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split 'provider:model_id' → ('provider', 'model_id').

    Ollama tags contain colons (`llama3:latest`) so split on the FIRST colon only.
    Bare strings without a colon are treated as Ollama models for backward compat.
    """
    if ":" not in spec:
        return ("ollama", spec)
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS:
        # unknown prefix — assume the whole thing is an ollama tag (e.g. "llama3:latest")
        return ("ollama", spec)
    return (provider, model)


def model_exists(provider: str, model_id: str) -> bool:
    available = list_models(provider)
    if not available:
        return False
    return model_id in available
