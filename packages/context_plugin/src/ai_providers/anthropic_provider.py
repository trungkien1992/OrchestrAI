import anthropic
from .base import AIProvider


class AnthropicProvider(AIProvider):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)

    async def complete(self, prompt: str, **kwargs) -> str:
        response = self.client.completions.create(
            prompt=prompt,
            model=kwargs.get("model", "claude-2"),
            max_tokens_to_sample=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
        )
        return response.completion.strip()

    async def chat(self, messages: list, **kwargs) -> str:
        # Anthropic API is prompt-based, not chat-based, so we join messages
        prompt = "\n".join([m["content"] for m in messages])
        return await self.complete(prompt, **kwargs)

    def validate_key(self) -> bool:
        try:
            self.client.completions.create(
                prompt="Hello", model="claude-2", max_tokens_to_sample=1
            )
            return True
        except Exception:
            return False
