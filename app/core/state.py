from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.agent.generation import AgentGeneration, build_generation
from app.mcp.config import McpConfigError, load_mcp_config

if TYPE_CHECKING:
    from app.agent.service import AgentService
    from app.core.config import Settings
    from app.core.structured_log import StructuredLogger
    from app.telegram.bot import TelegramBotService
    from app.telegram.queue import WebhookQueue


@dataclass
class ServiceState:
    settings: Settings | None = None
    logger: StructuredLogger | None = None
    telegram: TelegramBotService | None = None
    agent_service: AgentService | None = None
    generation: AgentGeneration | None = None
    generations: dict[str, AgentGeneration] = field(default_factory=dict)
    queue: WebhookQueue | None = None
    initialized: bool = False
    webhook_registered: bool = False
    reload_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    maintenance_task: asyncio.Task[None] | None = None

    def ready_reason(self) -> str | None:
        if not self.initialized:
            return "config_invalid"
        if self.queue is None or not self.queue.workers_running:
            return "workers_unavailable"
        if not self.webhook_registered:
            return "webhook_not_registered"
        if self.generation is None:
            return "mcp_tools_unavailable"
        enabled = [item for item in self.generation.config.servers if item.enabled]
        healthy = [item for item in self.generation.statuses.values() if item["status"] == "healthy"]
        if enabled and not healthy:
            return "mcp_tools_unavailable"
        return None


class ToolReloadError(RuntimeError):
    pass


async def reload_tools(service: ServiceState) -> dict[str, Any]:
    if service.settings is None or service.logger is None:
        raise ToolReloadError("clients_unavailable")
    async with service.reload_lock:
        try:
            config, tokens = load_mcp_config()
            candidate = await build_generation(service.settings, config, tokens, service.logger)
        except (McpConfigError, ValueError, RuntimeError) as exc:
            service.logger.error("tools_reload_failed", error=str(exc))
            raise ToolReloadError("mcp_config_invalid") from exc
        previous = service.generation
        service.generation = candidate
        service.generations[candidate.id] = candidate
        if previous is not None:
            previous.retired = True
        service.logger.info("tools_reloaded", servers=candidate.statuses)
        return {"status": "ok", "tool_count": len(candidate.tool_names), "servers": candidate.statuses}
