import google.generativeai as genai
from llm.base import BaseLLM
from config import GOOGLE_API_KEY, MODELS


class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = None):
        super().__init__(model_name or MODELS["gemini"])
        genai.configure(api_key=GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(model_name=self.model_name)

    def _call_api(self, system_prompt: str, user_message: str) -> str:
        full_prompt = f"{system_prompt}\n\n{user_message}"
        response = self._model.generate_content(full_prompt)
        return response.text
