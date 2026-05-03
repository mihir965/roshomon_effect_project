from google import genai
from google.genai import types

from llm.base import BaseLLM
from config import GOOGLE_API_KEY, MODELS


class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = None):
        super().__init__(model_name or MODELS["gemini"])
        self._client = genai.Client(api_key=GOOGLE_API_KEY)

    def _call_api(self, system_prompt: str, user_message: str) -> str:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=1024,
                temperature=0.7,
            ),
        )
        return response.text
