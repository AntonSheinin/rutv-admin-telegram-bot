from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    telegram_bot_token: str
    telegram_webhook_url: str
    telegram_webhook_secret: str
    telegram_admin_user_ids: set[int] = Field(validation_alias="TELEGRAM_ADMIN_USER_IDS")
    admin_api_token: str | None = None
    llm_model: str
    openai_api_key: str
    log_level: str = "INFO"
    max_tool_calls: int = Field(default=5, ge=1)
    request_timeout_seconds: int = Field(default=120, ge=1)
    update_worker_count: int = Field(default=2, ge=1)
    update_queue_max_size: int = Field(default=100, ge=1)
    update_dedupe_ttl_seconds: int = Field(default=86400, ge=1)
    max_tool_argument_bytes: int = Field(default=32768, ge=1)
    max_tool_result_bytes: int = Field(default=65536, ge=1)
    shutdown_drain_timeout_seconds: int = Field(default=30, ge=1)
    agent_tracing_enabled: bool = False

    @field_validator("telegram_admin_user_ids", mode="before")
    @classmethod
    def parse_admin_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return {int(part.strip()) for part in value.split(",") if part.strip()}
        if isinstance(value, int):
            return {value}
        return value

    @field_validator(
        "telegram_bot_token",
        "telegram_webhook_url",
        "telegram_webhook_secret",
        "llm_model",
        "openai_api_key",
    )
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("admin_api_token", mode="before")
    @classmethod
    def normalize_admin_api_token(cls, value: object) -> str | None | object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("telegram_admin_user_ids")
    @classmethod
    def require_admin_ids(cls, value: set[int]) -> set[int]:
        if not value:
            raise ValueError("must not be empty")
        return value

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        if env is None:
            return cls()
        return cls(_env_file=None, **{key.lower(): value for key, value in env.items()})
