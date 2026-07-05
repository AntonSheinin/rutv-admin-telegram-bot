import pytest

from app.core.config import ConfigError, Settings


def base_env():
    return {
        "TELEGRAM_BOT_TOKEN": "bot",
        "TELEGRAM_WEBHOOK_URL": "https://example.com/telegram/webhook",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
        "TELEGRAM_ADMIN_USER_IDS": "1,2",
        "MCP_SERVER_URL": "https://example.com/sse",
        "MCP_AUTH_TOKEN": "mcp",
        "OPENAI_API_KEY": "openai",
        "LLM_MODEL": "test-model",
    }


def test_settings_parse_defaults():
    settings = Settings.from_env(base_env())
    assert settings.telegram_admin_user_ids == {1, 2}
    assert settings.mcp_disabled_tools == set()
    assert settings.log_level == "INFO"
    assert settings.max_tool_calls == 5


def test_disabled_tools_parse():
    env = base_env()
    env["MCP_DISABLED_TOOLS"] = "delete_playlist, refresh_playlist"
    settings = Settings.from_env(env)
    assert settings.mcp_disabled_tools == {"delete_playlist", "refresh_playlist"}


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
