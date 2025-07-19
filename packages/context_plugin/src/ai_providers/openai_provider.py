import openai
from .base import AIProvider


class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        openai.api_key = api_key

    async def complete(self, prompt: str, **kwargs) -> str:
        response = openai.Completion.create(
            engine=kwargs.get("engine", "text-davinci-003"),
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
        )
        return response.choices[0].text.strip()

    async def chat(self, messages: list, **kwargs) -> str:
        response = openai.ChatCompletion.create(
            model=kwargs.get("model", "gpt-3.5-turbo"),
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
        )
        return response.choices[0].message["content"].strip()

    def validate_key(self) -> bool:
        try:
            openai.Model.list()
            return True
        except Exception:
            return False
