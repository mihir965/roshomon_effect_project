import requests
from llm.base import BaseLLM
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

CLOUD_PROVIDERS = {"openai", "anthropic", "gemini"}


class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = None):
        super().__init__(model_name or OLLAMA_MODEL)
        self._base_url = OLLAMA_BASE_URL

    def _call_api(self, system_prompt: str, user_message: str) -> str:
        response = requests.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def list_local_models() -> list[str]:
    """Return names of all models currently pulled in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return []


def is_ollama_model(name: str) -> bool:
    return name not in CLOUD_PROVIDERS
