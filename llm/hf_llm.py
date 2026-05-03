"""HuggingFace Inference API wrapper.

Uses huggingface_hub's InferenceClient, which speaks an OpenAI-compatible
chat-completions API across HF's hosted inference providers. Any instruct
model on the Hub that supports chat completion will work.
"""

from huggingface_hub import InferenceClient

from llm.base import BaseLLM
from config import HF_TOKEN


class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_name: str, provider: str | None = None):
        super().__init__(model_name)
        self._client = InferenceClient(
            model=model_name,
            token=HF_TOKEN or None,
            provider=provider,
        )

    def _call_api(self, system_prompt: str, user_message: str) -> str:
        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content
