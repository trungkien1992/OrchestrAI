import os
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .groq_provider import GroqProvider

PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
}


def get_provider(provider_name: str, api_key: str, **kwargs):
    provider_cls = PROVIDER_MAP.get(provider_name.lower())
    if not provider_cls:
        raise ValueError(f"Unknown provider: {provider_name}")
    return provider_cls(api_key, **kwargs)


def get_provider_from_env():
    provider_name = os.getenv("AI_PROVIDER", "openai")
    api_key = os.getenv("AI_API_KEY")
    if not api_key:
        raise ValueError("AI_API_KEY environment variable not set.")
    return get_provider(provider_name, api_key)
