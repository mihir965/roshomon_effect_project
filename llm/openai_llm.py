from openai import OpenAI
from llm.base import BaseLLM
from config import OPENAI_API_KEY, MODELS


class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str = None):
        super().__init__(model_name or MODELS["openai"])
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def _call_api(self, system_prompt: str, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
