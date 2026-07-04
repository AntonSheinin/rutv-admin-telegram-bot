import pytest

from app.policy import PolicyError, ToolPolicy
from app.openai_llm import OpenAILLMClient
from app.tool_schema import normalize_mcp_tool
from tests.test_config import base_env
from app.config import Settings


class Tool:
    name = "refresh_playlist"
    description = "Refresh playlist"
    inputSchema = {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}


def test_convert_tool_to_openai_schema():
    cached = normalize_mcp_tool(Tool())
    assert cached is not None
    client = OpenAILLMClient(Settings.from_env(base_env()))
    openai_tools = client.prepare_tools((cached,))
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["name"] == "refresh_playlist"


def test_convert_rejects_unsupported_schema_keywords():
    class UnsupportedTool:
        name = "bad"
        description = "Bad"
        inputSchema = {"type": "object", "oneOf": [{"type": "object"}]}

    cached = normalize_mcp_tool(UnsupportedTool())
    assert cached is not None
    client = OpenAILLMClient(Settings.from_env(base_env()))
    assert client.prepare_tools((cached,)) == []


def test_policy_rejects_disallowed_tool():
    policy = ToolPolicy({"allowed"}, False, 100)
    with pytest.raises(PolicyError):
        policy.validate_tool_call("blocked", {})


def test_policy_rejects_oversized_arguments():
    policy = ToolPolicy({"allowed"}, False, 5)
    with pytest.raises(PolicyError):
        policy.validate_tool_call("allowed", {"value": "too long"})
