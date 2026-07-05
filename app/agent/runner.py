from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Protocol

from app.core.structured_log import StructuredLogger
from app.core.config import Settings
from app.llm.client import LLMClient, LLMError, ToolCall, ToolResult
from app.mcp.cache import ToolCacheSnapshot
from app.mcp.client import McpClient, McpClientError
from app.mcp.policy import PolicyError, ToolPolicy


@dataclass(frozen=True)
class AgentResult:
    text: str
    request_id: str


class AgentRunner(Protocol):
    async def handle_message(
        self,
        text: str,
        tool_snapshot: ToolCacheSnapshot,
        *,
        update_id: int,
        user_id: int,
        chat_id: int,
    ) -> AgentResult:
        ...


class Agent:
    def __init__(
        self,
        settings: Settings,
        logger: StructuredLogger,
        llm: LLMClient,
        mcp_client: McpClient,
        policy: ToolPolicy,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._llm = llm
        self._mcp_client = mcp_client
        self._policy = policy

    async def handle_message(
        self,
        text: str,
        tool_snapshot: ToolCacheSnapshot,
        *,
        update_id: int,
        user_id: int,
        chat_id: int,
    ) -> AgentResult:
        request_id = str(uuid.uuid4())
        started = time.monotonic()
        self._logger.info(
            "agent_request_start",
            request_id=request_id,
            update_id=update_id,
            user_id=user_id,
            chat_id=chat_id,
        )
        try:
            result = await asyncio.wait_for(
                self._run(text, tool_snapshot, request_id=request_id, update_id=update_id, user_id=user_id, chat_id=chat_id),
                timeout=self._settings.request_timeout_seconds,
            )
            self._logger.info(
                "agent_request_finish",
                request_id=request_id,
                status="ok",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return AgentResult(text=result, request_id=request_id)
        except (asyncio.TimeoutError, LLMError, McpClientError, PolicyError) as exc:
            self._logger.error(
                "agent_request_finish",
                request_id=request_id,
                status="error",
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return AgentResult(text=f"No action was performed. Error: {exc}", request_id=request_id)

    async def _run(
        self,
        text: str,
        tool_snapshot: ToolCacheSnapshot,
        *,
        request_id: str,
        update_id: int,
        user_id: int,
        chat_id: int,
    ) -> str:
        if not tool_snapshot.is_valid:
            return "No action was performed. MCP tools are not available; try /reload_tools."

        conversation = self._llm.start_conversation(text)
        tools = self._llm.prepare_tools(tool_snapshot.tools)
        prepared_tool_names = {_tool_name(tool) for tool in tools}
        prepared_tool_names.discard(None)
        omitted_tool_names = sorted({tool.name for tool in tool_snapshot.tools} - prepared_tool_names)
        if omitted_tool_names:
            self._logger.info(
                "llm_provider_tools_omitted",
                request_id=request_id,
                provider=self._llm.provider,
                tool_names=omitted_tool_names,
            )
        if not tools:
            return "No action was performed. No tools are available for the configured LLM provider."
        allowed_tool_names = {name for name in prepared_tool_names if name is not None}
        tool_calls_used = 0

        async with AsyncExitStack() as stack:
            mcp_session = None
            while True:
                llm_response = await self._llm.create_response(conversation, tools)
                self._logger.debug(
                    "llm_response",
                    request_id=request_id,
                    provider=self._llm.provider,
                    model=self._llm.model,
                    tool_call_count=len(llm_response.tool_calls),
                )
                if not llm_response.tool_calls:
                    final_text = llm_response.text.strip()
                    return final_text or "No action was performed. Please provide a clearer request."

                self._llm.append_response(conversation, llm_response)
                for tool_call in llm_response.tool_calls:
                    tool_calls_used += 1
                    if tool_calls_used > self._settings.max_tool_calls:
                        return "No action was performed beyond the configured tool-call limit."
                    self._policy.validate_tool_call(tool_call.name, tool_call.arguments, allowed_tool_names)
                    if mcp_session is None:
                        mcp_session = await stack.enter_async_context(self._mcp_client.session())
                    self._llm.append_tool_result(
                        conversation,
                        tool_call,
                        await self._execute_tool_call(
                            mcp_session,
                            tool_call,
                            request_id=request_id,
                            update_id=update_id,
                            user_id=user_id,
                            chat_id=chat_id,
                        ),
                    )

    async def _execute_tool_call(
        self,
        mcp_session,
        tool_call: ToolCall,
        *,
        request_id: str,
        update_id: int,
        user_id: int,
        chat_id: int,
    ) -> ToolResult:
        self._logger.info(
            "mcp_tool_call_start",
            request_id=request_id,
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            update_id=update_id,
            user_id=user_id,
            chat_id=chat_id,
        )
        try:
            output = await asyncio.wait_for(
                mcp_session.call_tool(tool_call.name, tool_call.arguments),
                timeout=self._settings.tool_timeout_seconds,
            )
            status = "ok"
        except (asyncio.TimeoutError, McpClientError) as exc:
            output = f"ERROR: {exc}"
            status = "error"

        output, truncated = self._truncate_tool_result(output)
        self._logger.info(
            "mcp_tool_call_finish",
            request_id=request_id,
            tool_name=tool_call.name,
            status=status,
            truncated=truncated,
        )
        return ToolResult(status=status, result=output, truncated=truncated)

    def _truncate_tool_result(self, output: str) -> tuple[str, bool]:
        encoded = output.encode("utf-8", errors="replace")
        if len(encoded) <= self._settings.max_tool_result_bytes:
            return output, False
        truncated = encoded[: self._settings.max_tool_result_bytes].decode("utf-8", errors="replace")
        truncated += "\n[Output truncated before LLM summarization.]"
        return truncated, True


def _tool_name(tool: object) -> str | None:
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None
