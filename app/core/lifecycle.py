from __future__ import annotations

from app.agent.runner import Agent
from app.core.structured_log import StructuredLogger
from app.core.config import ConfigError, Settings
from app.core.state import ServiceState, ToolReloadError, reload_tools
from app.llm.factory import create_llm_client
from app.mcp.client import McpClient
from app.mcp.policy import ToolPolicy
from app.telegram.bot import TelegramBotService, TelegramServiceError
from app.telegram.handlers import process_update
from app.telegram.queue import WebhookQueue


async def startup(service: ServiceState) -> None:
    try:
        settings = Settings.from_env()
        logger = StructuredLogger(level=settings.log_level)
        policy = ToolPolicy(settings.mcp_disabled_tools, settings.max_tool_argument_bytes)
        telegram = TelegramBotService(settings)
        llm = create_llm_client(settings)
        mcp = McpClient(settings, logger)
        agent = Agent(settings, logger, llm, mcp, policy)
        service.settings = settings
        service.logger = logger
        service.policy = policy
        service.telegram = telegram
        service.llm = llm
        service.mcp = mcp
        service.agent = agent
        service.initialized = True
        service.queue = WebhookQueue(settings, logger, lambda update: process_update(service, update))
        await service.queue.start()
    except ConfigError as exc:
        service.logger = StructuredLogger()
        service.logger.error("startup_degraded", reason="config_invalid", error=str(exc))
        return
    except ValueError as exc:
        if service.logger is None:
            service.logger = StructuredLogger()
        service.logger.error("startup_degraded", reason="initialization_failed", error=str(exc))
        return

    try:
        await reload_tools(service)
    except ToolReloadError as exc:
        service.logger.warning("startup_degraded", reason="mcp_tools_unavailable", error=str(exc))
    try:
        await service.telegram.register_webhook()
        service.webhook_registered = True
        service.logger.info("telegram_webhook_registered")
    except TelegramServiceError as exc:
        service.webhook_registered = False
        service.logger.warning("startup_degraded", reason="webhook_registration_failed", error=str(exc))


async def shutdown(service: ServiceState) -> None:
    if service.queue is not None:
        await service.queue.shutdown()
    if service.telegram is not None:
        await service.telegram.close()
    if service.llm is not None:
        await service.llm.close()
    if service.logger is not None:
        service.logger.flush()
