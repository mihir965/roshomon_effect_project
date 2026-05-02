import anthropic
from llm.base import BaseLLM
from config import ANTHROPIC_API_KEY, MODELS


class AnthropicLLM(BaseLLM):
    def __init__(self, model_name: str = None):
        super().__init__(model_name or MODELS["anthropic"])
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def _call_api(self, system_prompt: str, user_message: str) -> str:
        message = self._client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text
