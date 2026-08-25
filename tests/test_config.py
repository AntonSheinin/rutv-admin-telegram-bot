import pytest
from pydantic import ValidationError

from app.core.config import Settings


def base_env() -> dict[str, str]:
    return {
        "TELEGRAM_BOT_TOKEN": "telegram-token",
        "TELEGRAM_WEBHOOK_URL": "https://example.test/webhook",
        "TELEGRAM_WEBHOOK_SECRET": "webhook-secret",
        "TELEGRAM_ADMIN_USER_IDS": "10,20",
        "ADMIN_API_TOKEN": "admin-token",
        "LLM_MODEL": "gpt-5-mini",
        "OPENAI_API_KEY": "openai-token",
    }


def test_settings_parse_global_infrastructure_values():
    settings = Settings.from_env(base_env())
    assert settings.telegram_admin_user_ids == {10, 20}
    assert settings.agent_tracing_enabled is False


def test_settings_accept_single_numeric_admin_id():
    env = base_env()
    env["TELEGRAM_ADMIN_USER_IDS"] = 10
    assert Settings.from_env(env).telegram_admin_user_ids == {10}


def test_settings_reject_missing_global_secret():
    env = base_env()
    del env["OPENAI_API_KEY"]
    with pytest.raises(ValidationError):
        Settings.from_env(env)


def test_settings_reject_empty_global_secret():
    env = base_env()
    env["OPENAI_API_KEY"] = " "
    with pytest.raises(ValidationError):
        Settings.from_env(env)


def test_settings_allow_missing_admin_api_token_for_telegram_only_operation():
    env = base_env()
    env["ADMIN_API_TOKEN"] = ""
    assert Settings.from_env(env).admin_api_token is None
