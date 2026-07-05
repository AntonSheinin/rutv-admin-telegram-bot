import pytest

from app.llm.openai import OpenAILLMClient
from app.mcp.policy import PolicyError, ToolPolicy
from app.mcp.schema import normalize_mcp_tool
from tests.test_config import base_env
from app.core.config import Settings


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


def test_normalize_missing_schema_defaults_to_empty_object_schema():
    class ToolWithoutSchema:
        name = "ping"
        description = "Ping"

    cached = normalize_mcp_tool(ToolWithoutSchema())

    assert cached is not None
    assert cached.input_schema == {"type": "object", "properties": {}, "additionalProperties": False}


def test_normalize_rejects_non_object_schema():
    class ToolWithArraySchema:
        name = "bad"
        description = "Bad"
        inputSchema = {"type": "array", "items": {"type": "string"}}

    assert normalize_mcp_tool(ToolWithArraySchema()) is None


def test_normalize_rejects_invalid_properties():
    class ToolWithInvalidProperties:
        name = "bad"
        description = "Bad"
        inputSchema = {"type": "object", "properties": []}

    assert normalize_mcp_tool(ToolWithInvalidProperties()) is None


def test_policy_rejects_disallowed_tool():
    policy = ToolPolicy({"blocked"}, 100)
    with pytest.raises(PolicyError):
        policy.validate_tool_call("blocked", {}, {"blocked"})


def test_policy_rejects_oversized_arguments():
    policy = ToolPolicy(set(), 5)
    with pytest.raises(PolicyError):
        policy.validate_tool_call("allowed", {"value": "too long"}, {"allowed"})


def test_policy_rejects_unavailable_tool():
    policy = ToolPolicy(set(), 100)
    with pytest.raises(PolicyError):
        policy.validate_tool_call("missing", {}, {"allowed"})


def test_policy_allows_tools_by_default():
    policy = ToolPolicy(set(), 100)
    assert policy.is_allowed("any_tool") is True
    assert policy.filter_tool_names(["a", "b"]) == {"a", "b"}


def test_policy_filters_disabled_tools():
    policy = ToolPolicy({"b"}, 100)
    assert policy.filter_tool_names(["a", "b", "c"]) == {"a", "c"}
