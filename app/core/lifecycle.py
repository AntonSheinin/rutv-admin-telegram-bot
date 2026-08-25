from __future__ import annotations

import asyncio

from pydantic import ValidationError

from app.agent.service import AgentService
from app.core.config import Settings
from app.core.structured_log import StructuredLogger
from app.core.state import ServiceState, ToolReloadError, reload_tools
from app.telegram.bot import TelegramBotService, TelegramServiceError
from app.telegram.handlers import process_update
from app.telegram.queue import WebhookQueue


async def startup(service: ServiceState) -> None:
    try:
        settings = Settings.from_env()
        service.settings = settings
        service.logger = StructuredLogger(level=settings.log_level)
        service.telegram = TelegramBotService(settings)
        service.agent_service = AgentService()
        service.initialized = True
        service.queue = WebhookQueue(settings, service.logger, lambda update: process_update(service, update))
        await service.queue.start()
    except (ValidationError, ValueError) as exc:
        service.logger = StructuredLogger()
        service.logger.error("startup_degraded", reason="config_invalid", error=str(exc))
        return

    try:
        await reload_tools(service)
    except ToolReloadError as exc:
        service.logger.warning("startup_degraded", reason="mcp_tools_unavailable", error=str(exc))
    try:
        await service.telegram.register_webhook()
        service.webhook_registered = True
    except TelegramServiceError as exc:
        service.logger.warning("startup_degraded", reason="webhook_registration_failed", error=str(exc))
    service.maintenance_task = asyncio.create_task(_maintenance(service))


async def _maintenance(service: ServiceState) -> None:
    while True:
        await asyncio.sleep(30)
        if service.agent_service is not None:
            await service.agent_service.purge_expired(service.generations)


async def shutdown(service: ServiceState) -> None:
    if service.maintenance_task is not None:
        service.maintenance_task.cancel()
        try:
            await service.maintenance_task
        except asyncio.CancelledError:
            pass
    if service.queue is not None:
        await service.queue.shutdown()
    if service.telegram is not None:
        await service.telegram.close()
    for generation in list(service.generations.values()):
        await generation.close()
    if service.logger is not None:
        service.logger.flush()
