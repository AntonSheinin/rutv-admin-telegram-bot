from pathlib import Path

import pytest

from app.mcp.config import McpConfigError, load_mcp_config


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "mcp_config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_enabled_server_and_secret_from_environment(tmp_path):
    path = write_config(tmp_path, '''[[servers]]
name = "bitrix24"
enabled = true
url = "https://mcp.bitrix24.com/mcp/"
disabled_tools = ["delete_task"]
confirmation_required_tools = ["task_add"]
''')
    config, tokens = load_mcp_config({"MCP_BITRIX24_AUTH_TOKEN": "secret"}, path)
    assert config.servers[0].token_env_name == "MCP_BITRIX24_AUTH_TOKEN"
    assert tokens == {"bitrix24": "secret"}


def test_enabled_server_requires_derived_secret(tmp_path):
    path = write_config(tmp_path, '[[servers]]\nname = "playlist"\nenabled = true\nurl = "https://example.com/mcp"\n')
    with pytest.raises(McpConfigError, match="MCP_PLAYLIST_AUTH_TOKEN"):
        load_mcp_config({}, path)


def test_tenant_settings_must_be_configured_together(tmp_path):
    path = write_config(tmp_path, '[[servers]]\nname = "playlist"\nenabled = false\nurl = "https://example.com/mcp"\nexpected_tenant = "rutv"\n')
    with pytest.raises(McpConfigError, match="required together"):
        load_mcp_config({}, path)


def test_accepts_bitrix24_access_token_authentication(tmp_path):
    path = write_config(
        tmp_path,
        '[[servers]]\nname = "bitrix24"\nenabled = true\nurl = "https://mcp.bitrix24.com/mcp/"\nauth_type = "access_token"\n',
    )
    config, _ = load_mcp_config({"MCP_BITRIX24_AUTH_TOKEN": "secret"}, path)
    assert config.servers[0].auth_type == "access_token"


def test_explicit_environment_overrides_dotenv_lookup(tmp_path):
    path = write_config(tmp_path, '[[servers]]\nname = "playlist"\nenabled = true\nurl = "https://example.com/mcp"\n')
    _, tokens = load_mcp_config({"MCP_PLAYLIST_AUTH_TOKEN": "explicit"}, path)
    assert tokens["playlist"] == "explicit"
