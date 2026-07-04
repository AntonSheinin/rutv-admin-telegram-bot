from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent import AgentRunner
from app.audit import AuditLogger
from app.config import Settings
from app.llm import LLMClient
from app.mcp_client import McpClient, McpClientError, ToolCache
from app.policy import ToolPolicy
from app.telegram_bot import TelegramBotService
from app.webhook_queue import WebhookQueue


@dataclass
class ServiceState:
    settings: Settings | None = None
    audit: AuditLogger | None = None
    telegram: TelegramBotService | None = None
    llm: LLMClient | None = None
    mcp: McpClient | None = None
    policy: ToolPolicy | None = None
    tool_cache: ToolCache = field(default_factory=ToolCache)
    agent: AgentRunner | None = None
    queue: WebhookQueue | None = None
    config_valid: bool = False
    webhook_registered: bool = False
    startup_error: str | None = None

    def ready_reason(self) -> str | None:
        if not self.config_valid:
            return "config_invalid"
        if self.queue is None or not self.queue.workers_running:
            return "workers_unavailable"
        if not self.webhook_registered:
            return "webhook_not_registered"
        if not self.tool_cache.is_valid:
            return self.tool_cache.last_error or "mcp_tools_unavailable"
        if not all([self.telegram, self.llm, self.mcp, self.policy, self.agent]):
            return "clients_unavailable"
        return None


class ToolReloadError(RuntimeError):
    pass


async def reload_tools(service: ServiceState) -> dict[str, Any]:
    if not service.mcp or not service.policy:
        raise ToolReloadError("clients_unavailable")
    try:
        snapshot = await service.mcp.load_tools(service.policy)
        await service.tool_cache.replace(snapshot)
        if service.audit:
            service.audit.info("tools_reloaded", tool_count=len(snapshot.tools))
        return {"status": "ok", "tool_count": len(snapshot.tools)}
    except McpClientError as exc:
        await service.tool_cache.mark_error(str(exc))
        if service.audit:
            service.audit.error("tools_reload_failed", error=str(exc))
        raise ToolReloadError("mcp_tools_unavailable") from exc
