from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.mcp.cache import ToolCache
from app.mcp.client import McpClientError

if TYPE_CHECKING:
    from app.agent.runner import AgentRunner
    from app.core.config import Settings
    from app.core.structured_log import StructuredLogger
    from app.llm.client import LLMClient
    from app.mcp.client import McpClient
    from app.mcp.policy import ToolPolicy
    from app.telegram.bot import TelegramBotService
    from app.telegram.queue import WebhookQueue


@dataclass
class ServiceState:
    settings: Settings | None = None
    logger: StructuredLogger | None = None
    telegram: TelegramBotService | None = None
    llm: LLMClient | None = None
    mcp: McpClient | None = None
    policy: ToolPolicy | None = None
    tool_cache: ToolCache = field(default_factory=ToolCache)
    agent: AgentRunner | None = None
    queue: WebhookQueue | None = None
    initialized: bool = False
    webhook_registered: bool = False

    def ready_reason(self) -> str | None:
        if not self.initialized:
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
        service.tool_cache.replace(snapshot)
        if service.logger:
            service.logger.info("tools_reloaded", tool_count=len(snapshot.tools))
        return {"status": "ok", "tool_count": len(snapshot.tools)}
    except McpClientError as exc:
        service.tool_cache.mark_error(str(exc))
        if service.logger:
            service.logger.error("tools_reload_failed", error=str(exc))
        raise ToolReloadError("mcp_tools_unavailable") from exc
