from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(ValueError):
    pass


def _get_required(env: dict[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required")
    return value


def _get_int(env: dict[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = env.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _get_bool(env: dict[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean")


def _parse_admin_ids(raw: str) -> set[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ConfigError("TELEGRAM_ADMIN_USER_IDS must not be empty")
    result: set[int] = set()
    for value in values:
        try:
            result.add(int(value))
        except ValueError as exc:
            raise ConfigError("TELEGRAM_ADMIN_USER_IDS must contain integers") from exc
    return result


def _parse_tool_names(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    telegram_webhook_url: str
    telegram_webhook_secret: str
    telegram_admin_user_ids: set[int]
    mcp_server_url: str
    mcp_auth_token: str
    mcp_allowed_tools: set[str]
    admin_api_token: str
    environment: str = "production"
    llm_provider: str = "openai"
    llm_model: str = ""
    openai_api_key: str | None = None
    audit_log_path: str = ""
    log_level: str = "INFO"
    mcp_allow_all_tools: bool = False
    enable_agent_http: bool = False
    max_tool_calls: int = 5
    request_timeout_seconds: int = 120
    tool_timeout_seconds: int = 30
    max_concurrent_requests: int = 4
    update_worker_count: int = 2
    update_queue_max_size: int = 100
    update_dedupe_ttl_seconds: int = 86400
    max_tool_argument_bytes: int = 32768
    max_tool_result_bytes: int = 65536
    shutdown_drain_timeout_seconds: int = 30

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        source = dict(os.environ if env is None else env)
        environment = source.get("ENVIRONMENT", "production").strip() or "production"
        mcp_allow_all_tools = _get_bool(source, "MCP_ALLOW_ALL_TOOLS", False)
        if mcp_allow_all_tools and environment != "local":
            raise ConfigError("MCP_ALLOW_ALL_TOOLS=true is only valid when ENVIRONMENT=local")

        raw_allowed_tools = source.get("MCP_ALLOWED_TOOLS", "")
        allowed_tools = _parse_tool_names(raw_allowed_tools)
        if not allowed_tools and not mcp_allow_all_tools:
            raise ConfigError("MCP_ALLOWED_TOOLS must not be empty unless MCP_ALLOW_ALL_TOOLS=true")

        llm_provider = source.get("LLM_PROVIDER", "openai").strip() or "openai"
        llm_model = source.get("LLM_MODEL", "").strip()
        if not llm_model:
            raise ConfigError("LLM_MODEL is required")
        openai_api_key = source.get("OPENAI_API_KEY", "").strip() or None
        if llm_provider == "openai" and not openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        return cls(
            telegram_bot_token=_get_required(source, "TELEGRAM_BOT_TOKEN"),
            telegram_webhook_url=_get_required(source, "TELEGRAM_WEBHOOK_URL"),
            telegram_webhook_secret=_get_required(source, "TELEGRAM_WEBHOOK_SECRET"),
            telegram_admin_user_ids=_parse_admin_ids(_get_required(source, "TELEGRAM_ADMIN_USER_IDS")),
            mcp_server_url=_get_required(source, "MCP_SERVER_URL"),
            mcp_auth_token=_get_required(source, "MCP_AUTH_TOKEN"),
            mcp_allowed_tools=allowed_tools,
            admin_api_token=_get_required(source, "ADMIN_API_TOKEN"),
            environment=environment,
            llm_provider=llm_provider,
            llm_model=llm_model,
            openai_api_key=openai_api_key,
            audit_log_path=source.get("AUDIT_LOG_PATH", "").strip(),
            log_level=source.get("LOG_LEVEL", "INFO").strip().upper() or "INFO",
            mcp_allow_all_tools=mcp_allow_all_tools,
            enable_agent_http=_get_bool(source, "ENABLE_AGENT_HTTP", False),
            max_tool_calls=_get_int(source, "MAX_TOOL_CALLS", 5, 1),
            request_timeout_seconds=_get_int(source, "REQUEST_TIMEOUT_SECONDS", 120, 1),
            tool_timeout_seconds=_get_int(source, "TOOL_TIMEOUT_SECONDS", 30, 1),
            max_concurrent_requests=_get_int(source, "MAX_CONCURRENT_REQUESTS", 4, 1),
            update_worker_count=_get_int(source, "UPDATE_WORKER_COUNT", 2, 1),
            update_queue_max_size=_get_int(source, "UPDATE_QUEUE_MAX_SIZE", 100, 1),
            update_dedupe_ttl_seconds=_get_int(source, "UPDATE_DEDUPE_TTL_SECONDS", 86400, 1),
            max_tool_argument_bytes=_get_int(source, "MAX_TOOL_ARGUMENT_BYTES", 32768, 1),
            max_tool_result_bytes=_get_int(source, "MAX_TOOL_RESULT_BYTES", 65536, 1),
            shutdown_drain_timeout_seconds=_get_int(source, "SHUTDOWN_DRAIN_TIMEOUT_SECONDS", 30, 1),
        )
