from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    discord_bot_token: str

    anthropic_api_key: str = ""
    openai_api_key: str = ""

    database_url: str = "sqlite+aiosqlite:///./data/bot.db"

    default_provider: str = "anthropic"
    default_model: str = "claude-opus-4-8"

    max_tool_iterations: int = 10
    max_conversation_history: int = 20

    # Anthropic extended thinking. Disabled automatically for small-budget calls
    # (e.g. the automod classifier) so reasoning tokens can't starve the response.
    anthropic_extended_thinking: bool = False

    # Per-user rate limits (token-bucket, keyed on (guild_id, user_id)).
    ask_rate_limit: int = 5          # max /ask + @mention LLM calls...
    ask_rate_window: float = 60.0    # ...per this many seconds
    automod_rate_limit: int = 15     # max automod classifications per user...
    automod_rate_window: float = 60.0  # ...per this many seconds


settings = Settings()
