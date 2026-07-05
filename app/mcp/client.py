from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

from app.core.structured_log import StructuredLogger
from app.core.config import Settings
from app.mcp.cache import ToolCacheSnapshot
from app.mcp.policy import ToolPolicy
from app.mcp.schema import normalize_mcp_tool


class McpClientError(RuntimeError):
    pass


class McpClient:
    def __init__(self, settings: Settings, logger: StructuredLogger) -> None:
        self._settings = settings
        self._logger = logger
        self._headers = {"Authorization": f"Bearer {settings.mcp_auth_token}"}

    def session(self) -> "McpSession":
        return McpSession(self)

    async def load_tools(self, policy: ToolPolicy) -> ToolCacheSnapshot:
        try:
            async with self.session() as session:
                response = await session.list_tools()
            raw_tools = list(getattr(response, "tools", []) or [])
        except Exception as exc:
            raise McpClientError(f"Failed to load MCP tools: {exc}") from exc

        allowed_names = policy.filter_tool_names([getattr(tool, "name", "") for tool in raw_tools])
        cached = []
        for tool in raw_tools:
            if getattr(tool, "name", None) not in allowed_names:
                continue
            converted = normalize_mcp_tool(tool)
            if converted is None:
                self._logger.debug("mcp_tool_schema_omitted", tool_name=getattr(tool, "name", None))
                continue
            cached.append(converted)
        if not cached:
            raise McpClientError("No usable MCP tools after filtering/conversion")
        return ToolCacheSnapshot(tuple(cached))

    def _stringify_result(self, result: Any) -> str:
        structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
        if structured is not None:
            return json.dumps(structured, ensure_ascii=True, default=str)
        content = getattr(result, "content", None)
        if content is not None:
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(result)


class McpSession:
    def __init__(self, client: McpClient) -> None:
        self._client = client
        self._stack: AsyncExitStack | None = None
        self._session = None

    async def __aenter__(self) -> "McpSession":
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except Exception as exc:  # pragma: no cover - depends on installed package
            raise McpClientError("Official Python MCP SDK is not available") from exc

        stack = AsyncExitStack()
        try:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(self._client._settings.mcp_server_url, headers=self._client._headers)
            )
            self._session = await stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
            self._stack = stack
            return self
        except Exception:
            await stack.aclose()
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self._session is None:
            raise McpClientError("MCP session is not open")
        try:
            result = await self._session.call_tool(tool_name, arguments)
            return self._client._stringify_result(result)
        except Exception as exc:
            raise McpClientError(f"MCP tool call failed: {exc}") from exc
