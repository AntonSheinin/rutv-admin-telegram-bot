from contextlib import asynccontextmanager

import pytest

from app.core.config import Settings
from app.core.structured_log import StructuredLogger
from app.mcp.client import McpClient
from tests.test_config import base_env


@pytest.mark.asyncio
async def test_mcp_session_uses_streamable_http_transport(monkeypatch):
    calls = []

    @asynccontextmanager
    async def fake_streamable_client(url, headers):
        calls.append({"url": url, "headers": headers})
        yield "read", "write", lambda: "session-id"

    class FakeClientSession:
        def __init__(self, read, write):
            self.read = read
            self.write = write

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            return None

    monkeypatch.setattr("mcp.client.streamable_http.streamablehttp_client", fake_streamable_client)
    monkeypatch.setattr("mcp.ClientSession", FakeClientSession)

    env = base_env()
    env["MCP_SERVER_URL"] = "http://playlist.example:8090/mcp"
    env["MCP_AUTH_TOKEN"] = "mcp-secret"
    client = McpClient(Settings.from_env(env), StructuredLogger())

    async with client.session():
        pass

    assert calls == [
        {
            "url": "http://playlist.example:8090/mcp",
            "headers": {"Authorization": "Bearer mcp-secret"},
        }
    ]
