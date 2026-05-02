from openai import OpenAI
from llm.base import BaseLLM
from config import GROQ_API_KEY, GROQ_MODELS


class GroqLLM(BaseLLM):
    def __init__(self, model_key: str):
        model_id = GROQ_MODELS[model_key]
        super().__init__(model_id)
        self._client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )

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
