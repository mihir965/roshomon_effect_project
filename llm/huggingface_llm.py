from huggingface_hub import InferenceClient
from llm.base import BaseLLM
from config import HF_API_KEY, HF_MODELS


class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_key: str):
        model_id, provider = HF_MODELS[model_key]
        super().__init__(model_id)
        self._client = InferenceClient(model=model_id, token=HF_API_KEY, provider=provider)

    def _call_api(self, system_prompt: str, user_message: str) -> str:
        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
