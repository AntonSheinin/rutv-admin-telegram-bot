from dataclasses import dataclass

import pytest

from app.agent.runner import Agent
from app.core.structured_log import StructuredLogger
from app.core.config import Settings
from app.llm.client import LLMResponse, ToolCall
from app.mcp.cache import ToolCacheSnapshot
from app.mcp.policy import ToolPolicy
from app.mcp.schema import CachedTool

from tests.test_config import base_env


def settings() -> Settings:
    return Settings.from_env(base_env())


def snapshot() -> ToolCacheSnapshot:
    tool = CachedTool(
        name="refresh_playlist",
        description="Refresh",
        input_schema={"type": "object", "properties": {}},
    )
    return ToolCacheSnapshot((tool,))


def snapshot_with_omitted_tool() -> ToolCacheSnapshot:
    return ToolCacheSnapshot(
        (
            CachedTool(
                name="refresh_playlist",
                description="Refresh",
                input_schema={"type": "object", "properties": {}},
            ),
            CachedTool(
                name="omitted_tool",
                description="Omitted",
                input_schema={"type": "object", "properties": {}},
            ),
        )
    )


class DirectLLM:
    provider = "fake"
    model = "fake-model"

    def prepare_tools(self, tools):
        return [{"name": tool.name} for tool in tools]

    def start_conversation(self, user_text):
        return [{"role": "user", "content": user_text}]

    async def create_response(self, input_items, tools):
        return LLMResponse(text="Done without tools.", tool_calls=[])

    def append_response(self, conversation, response):
        pass

    def append_tool_result(self, conversation, tool_call, result):
        conversation.append({"tool": tool_call.name, "result": result.result})


class ToolThenFinalLLM:
    provider = "fake"
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def prepare_tools(self, tools):
        return [{"name": tool.name} for tool in tools]

    def start_conversation(self, user_text):
        return [{"role": "user", "content": user_text}]

    async def create_response(self, input_items, tools):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(call_id="call_1", name="refresh_playlist", arguments={})],
                provider_state=[{"type": "function_call", "call_id": "call_1", "name": "refresh_playlist", "arguments": "{}"}],
            )
        return LLMResponse(text="Refresh completed.", tool_calls=[])

    def append_response(self, conversation, response):
        conversation.extend(response.provider_state or [])

    def append_tool_result(self, conversation, tool_call, result):
        conversation.append({"tool": tool_call.name, "result": result.result})


class MissingToolLLM(ToolThenFinalLLM):
    async def create_response(self, input_items, tools):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(call_id="call_1", name="missing_tool", arguments={})],
                provider_state=[{"type": "function_call", "call_id": "call_1", "name": "missing_tool", "arguments": "{}"}],
            )
        return LLMResponse(text="Should not get here.", tool_calls=[])


class OmittedPreparedToolLLM(ToolThenFinalLLM):
    def prepare_tools(self, tools):
        return [{"name": "refresh_playlist"}]

    async def create_response(self, input_items, tools):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(call_id="call_1", name="omitted_tool", arguments={})],
                provider_state=[{"type": "function_call", "call_id": "call_1", "name": "omitted_tool", "arguments": "{}"}],
            )
        return LLMResponse(text="Should not get here.", tool_calls=[])


class FakeMcpSession:
    def __init__(self, parent):
        self.parent = parent

    async def __aenter__(self):
        self.parent.open_count += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.parent.close_count += 1

    async def call_tool(self, name, arguments):
        self.parent.call_count += 1
        return "ok"


class FakeMcpClient:
    def __init__(self):
        self.open_count = 0
        self.close_count = 0
        self.call_count = 0

    def session(self):
        return FakeMcpSession(self)


@pytest.mark.asyncio
async def test_agent_direct_answer_does_not_open_mcp_session():
    fake_mcp = FakeMcpClient()
    agent = Agent(settings(), StructuredLogger(), DirectLLM(), fake_mcp, ToolPolicy(set(), 1000))

    result = await agent.handle_message("hello", snapshot(), update_id=1, user_id=1, chat_id=1)

    assert result.text == "Done without tools."
    assert fake_mcp.open_count == 0


@pytest.mark.asyncio
async def test_agent_reuses_one_mcp_session_for_tool_loop():
    fake_mcp = FakeMcpClient()
    agent = Agent(settings(), StructuredLogger(), ToolThenFinalLLM(), fake_mcp, ToolPolicy(set(), 1000))

    result = await agent.handle_message("refresh", snapshot(), update_id=1, user_id=1, chat_id=1)

    assert result.text == "Refresh completed."
    assert fake_mcp.open_count == 1
    assert fake_mcp.close_count == 1
    assert fake_mcp.call_count == 1


@pytest.mark.asyncio
async def test_agent_rejects_tool_not_in_snapshot():
    fake_mcp = FakeMcpClient()
    agent = Agent(settings(), StructuredLogger(), MissingToolLLM(), fake_mcp, ToolPolicy(set(), 1000))

    result = await agent.handle_message("refresh", snapshot(), update_id=1, user_id=1, chat_id=1)

    assert "Tool is not available: missing_tool" in result.text
    assert fake_mcp.call_count == 0


@pytest.mark.asyncio
async def test_agent_rejects_tool_not_prepared_for_provider():
    fake_mcp = FakeMcpClient()
    agent = Agent(settings(), StructuredLogger(), OmittedPreparedToolLLM(), fake_mcp, ToolPolicy(set(), 1000))

    result = await agent.handle_message("refresh", snapshot_with_omitted_tool(), update_id=1, user_id=1, chat_id=1)

    assert "Tool is not available: omitted_tool" in result.text
    assert fake_mcp.call_count == 0
