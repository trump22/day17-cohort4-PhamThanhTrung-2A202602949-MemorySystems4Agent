from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float = 0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value):
    value = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "openai-compatible": "custom",
    }.get(value.strip().lower(), value.strip().lower())
    if value not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported provider: {value}")
    return value


def build_chat_model(config):
    p = normalize_provider(config.provider)
    kw = dict(model=config.model_name, temperature=config.temperature)
    if p != "ollama" and not config.api_key:
        raise ValueError(f"Missing API key for {p}")
    if config.api_key:
        kw["api_key"] = config.api_key
    if config.base_url:
        kw["base_url"] = config.base_url
    if p in {"openai", "custom", "openrouter"}:
        from langchain_openai import ChatOpenAI

        if p == "custom" and not config.base_url:
            raise ValueError("CUSTOM_BASE_URL required")
        if p == "openrouter":
            kw["base_url"] = config.base_url or "https://openrouter.ai/api/v1"
        return ChatOpenAI(**kw)
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**kw)
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**kw)
    from langchain_ollama import ChatOllama

    return ChatOllama(**kw)
