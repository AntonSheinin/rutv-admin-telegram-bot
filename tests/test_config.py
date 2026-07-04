import pytest

from app.config import ConfigError, Settings


def base_env():
    return {
        "TELEGRAM_BOT_TOKEN": "bot",
        "TELEGRAM_WEBHOOK_URL": "https://example.com/telegram/webhook",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
        "TELEGRAM_ADMIN_USER_IDS": "1,2",
        "MCP_SERVER_URL": "https://example.com/sse",
        "MCP_AUTH_TOKEN": "mcp",
        "MCP_ALLOWED_TOOLS": "a,b",
        "ADMIN_API_TOKEN": "admin",
        "OPENAI_API_KEY": "openai",
        "LLM_MODEL": "test-model",
    }


def test_settings_parse_defaults():
    settings = Settings.from_env(base_env())
    assert settings.telegram_admin_user_ids == {1, 2}
    assert settings.mcp_allowed_tools == {"a", "b"}
    assert settings.environment == "production"
    assert settings.log_level == "INFO"
    assert settings.max_tool_calls == 5


def test_allow_all_tools_only_local():
    env = base_env()
    env["MCP_ALLOWED_TOOLS"] = ""
    env["MCP_ALLOW_ALL_TOOLS"] = "true"
    with pytest.raises(ConfigError):
        Settings.from_env(env)
    env["ENVIRONMENT"] = "local"
    settings = Settings.from_env(env)
    assert settings.mcp_allow_all_tools is True


def test_admin_ids_required_and_integer():
    env = base_env()
    env["TELEGRAM_ADMIN_USER_IDS"] = "abc"
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_llm_model_required():
    env = base_env()
    env["LLM_MODEL"] = ""
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_non_openai_provider_does_not_require_openai_key():
    env = base_env()
    env["LLM_PROVIDER"] = "custom"
    env["OPENAI_API_KEY"] = ""
    settings = Settings.from_env(env)
    assert settings.llm_provider == "custom"
